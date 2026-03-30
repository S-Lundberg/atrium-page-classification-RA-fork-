import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#import torch
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from riksarkivet_prompt_from_meta import tokenize_safe_single, create_prompt
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torchvision
from torchvision import transforms
from tqdm import tqdm
import clip
from PIL import Image, ImageEnhance, ImageFilter

from huggingface_hub import PyTorchModelHubMixin

Image.MAX_IMAGE_PIXELS = 700_000_000
import string
import torch.nn as nn

from minor_classes import *
from utils import main_class_report

def collate_with_prompts(batch):
    images = torch.stack([b[0] for b in batch])
    classes = torch.tensor([b[1] for b in batch])
    prompt_lists = [b[2] for b in batch]

    flat_prompts = [p for lst in prompt_lists for p in lst]
    #counts = [len(lst) for lst in prompt_lists]

    return images, classes, flat_prompts


class CLIP(nn.Module, PyTorchModelHubMixin):
    # --- HF‐Hub Mixin metadata (used in the generated model card) ---
    library_name = "ufal/clip-historical-page"
    tags          = ["vision-language", "clip", "custom"]
    pipeline_tag  = "few-shot-image-classification"

    def __init__(self,
                 max_category_samples: int | None,
                 eval_max_category_samples: int | None,
                 top_N: int,
                 model_name: str,
                 model_revision: str | None,
                 device: str,
                 seed: int,
                 categories_tsv: str,
                 categories_dir: str = "./category_descriptions",
                 test_ratio: float = 0.1,
                 input_format: str = "jpeg",
                 output_dir: str = None,
                 model_dir: str = None,
                 cp_dir: str = None,
                 cat_prefix: str = None,
                 safety_check: bool = True,
                 avg: bool = False,
                 zero_shot: bool = False,
                 prompt_path: str | None = None,
                 one_2_one: bool = False,
                 ensemble: bool = False,
                 metadata_path: str | None = None,
                 plot_embeddings: bool = False,
                 advanced_split: bool = False
                 ):
        super().__init__()  # initialize nn.Module
        # all your existing init logic follows unchanged:
        self.upper_category_limit      = max_category_samples
        self.upper_category_limit_eval = eval_max_category_samples
        self.top_N                     = top_N
        self.seed                      = seed
        self.avg                       = avg
        self.device                    = device
        self.zero_shot                 = zero_shot
        self.prompt_path               = prompt_path
        self.one_2_one                 = one_2_one
        self.test_fraction = test_ratio
        self.file_format = input_format
        self.safe_load = safety_check
        self.ensemble = ensemble
        self.metadata_path = metadata_path
        self.plot_embeddings = plot_embeddings
        self.advanced_split = advanced_split
        print(f"Chosen file handling:\t{self.file_format} format \n\tSafety load of images:\t{self.safe_load}")
        if self.one_2_one and self.ensemble:
            raise ValueError("one_2_one and ensemble can't be True at the same time.")

        self.output_dir = Path(__file__).parent / "result" if output_dir is None else Path(output_dir)
        self.models_dir = Path(__file__).parent / "models" if model_dir is None else Path(model_dir)
        self.checkpoints_dir = Path(__file__).parent / "model_checkpoints" if cp_dir is None else Path(cp_dir)
        self.download_root = '/lnet/work/projects/atrium/cache/clip'

        self.model_code_name = f'{model_name.replace("/", "").replace("@", "-")}_{model_revision}'

        # Must set jit=False for training
        self.model, self.preprocess = clip.load(model_name, device=device,
                                                download_root=self.download_root, jit=False)

        image_size = (self.preprocess.transforms[0].size, self.preprocess.transforms[0].size)
        image_mean = self.preprocess.transforms[-1].mean
        image_std = self.preprocess.transforms[-1].std

        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))),
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2))))
            ], p=0.5),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

        self.eval_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])
        if self.ensemble:
            prompt_tsv = Path(self.prompt_path)
            out_path = Path("category_descriptions") / "prompts"
            if not prompt_tsv.exists():
                print("No prompt file found for ensemble mode.")
                print("Generating prompts using metadata and category templates...")
                create_prompt(
                    metadata_csv_path=metadata_path,
                    category_tsv_path=categories_tsv,
                    out_path = out_path
                )
            else:
                print(f"Using existing prompt file: {prompt_tsv}")

            # Load prompts per file
            print("Loading prompts into memory...")
            df = pd.read_csv(prompt_tsv, sep="\t")

            self.prompts_by_file = (
                df.groupby("filename")["prompt"]
                .apply(list)
                .to_dict()
            )

            print(f"Loaded prompts for {len(self.prompts_by_file)} files.")


        if self.avg:
            loaded_cats = load_categories(categories_tsv, prefix=cat_prefix, directory=categories_dir)
            self.categories = sorted(loaded_cats.keys())
            self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
            self.texts = loaded_cats  # dict of lists
            print(f"Categories: {self.categories} with multiple descriptions per category.")
            self.text_features = self._get_averaged_text_features()
            for cat, descs in self.texts.items():
                print(f"Category: {cat}, Descriptions:")
                for desc in descs:
                    print(f"  - {desc}")
        else:
            loaded_cats = load_categories(categories_tsv, directory=categories_dir)
            if type(loaded_cats) is dict:
                self.categories = sorted(loaded_cats.keys())
                self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
                self.texts = loaded_cats  # dict of lists
                self.text_features = self._get_averaged_text_features()
                self.avg = True
            else:
                self.categories = [label for label, desc in loaded_cats]
                self.texts = [desc for label, desc in loaded_cats]
                self.text_inputs = torch.cat([clip.tokenize(f"a scan of {description}") for description in self.texts]).to(
                        device)
                
            print(f"Categories: {self.categories} with single description per category.")
            if isinstance(self.texts, list):
                for cat, desc in zip(self.categories, self.texts):
                    print(f"Category: {cat}, Description: {desc}")
            else:
                for cat, descs in self.texts.items():
                    print(f"Category: {cat}, Descriptions:")
                    for desc in descs:
                        print(f"  - {desc}")
        if self.one_2_one and self.prompt_path is not None:
            df = pd.read_csv(self.prompt_path, sep="\t")

            # filename → prompt
            self.prompt_map = {
                row["filename"]: row["prompt"]
                for _, row in df.iterrows()
            }

            print(f"Loaded {len(self.prompt_map)} one-to-one prompts from {self.prompt_path}.")

        self.model_name = model_name

        self.num_prediction_classes = len(self.categories)
        print(f"\tNumber of prediction classes: {self.num_prediction_classes}")
        print(f"\tModel name: {self.model_name}")
        print(f"\tModel code name: {self.model_code_name}")

        self.categories_dir = categories_dir
        self.categories_tsv = categories_tsv
        self.metadata_path  = metadata_path
    def aggregate_prompts(self, prompts, strategy="mean"):
        if len(prompts) == 0:
            D = self.model.text_projection.shape[1]
            return torch.zeros(D, device=self.device)

        tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        if strategy == "mean":
            out = feats.mean(dim=0)
        else:
            out, _ = feats.max(dim=0)

        return out / out.norm()



    def _get_averaged_text_features_old(self):
        """Computes averaged text features for each category."""
        all_features = []
        with torch.no_grad():
            for category in self.categories:
                descriptions = [f"a scan of {desc}" for desc in self.texts[category]]
                #print(descriptions)
                tokens = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                mean_features = features.mean(dim=0)
                mean_features /= mean_features.norm()
                all_features.append(mean_features)
        return torch.stack(all_features)

    def _get_averaged_text_features(self):
        """Computes averaged text features for each category and plots 1D heatmaps per prompt + mean."""
        all_features = []

        with torch.no_grad():
            for category in self.categories:
                descriptions = [f"a scan of {desc}" for desc in self.texts[category]]
                tokens = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)

                # Encode prompts
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)

                # Average embedding
                mean_features = features.mean(dim=0)
                mean_features = mean_features / mean_features.norm()

                # --- SUBPLOTS: one per prompt + mean ---
                num_prompts = features.shape[0]
                total_plots = num_prompts + 1  # prompts + mean

                fig, axes = plt.subplots(total_plots, 1, figsize=(10, 1.2 * total_plots), sharex=True)

                for i in range(num_prompts):
                    axes[i].imshow(features[i].unsqueeze(0).cpu().numpy(),
                                aspect="auto", cmap="viridis")
                    axes[i].set_ylabel(f"p{i}", rotation=0, labelpad=20)
                    axes[i].set_yticks([])

                # Mean embedding subplot
                axes[-1].imshow(mean_features.unsqueeze(0).cpu().numpy(),
                                aspect="auto", cmap="viridis")
                axes[-1].set_ylabel("mean", rotation=0, labelpad=20)
                axes[-1].set_yticks([])
                os.makedirs(self.output_dir / "embeddings", exist_ok=True)
                plt.suptitle(f"Embedding dimensions – {category}")
                plt.xlabel("embedding dimension")
                plt.tight_layout()
                plt.savefig(self.output_dir / "embeddings" / f"{self.model_code_name}_{category}_embedding_heatmap.png")
                # --- END SUBPLOTS ---

                all_features.append(mean_features)

        return torch.stack(all_features)

    def train(self, train_dir: str, log_dir: str, eval_dir: str = None,
              num_epochs: int = 5, batch_size: int = 8, learning_rate: float = 1e-7):
        """
        Fine-tunes the CLIP model based on the provided training and evaluation directories.
        """
        print("\t*\tStarting CLIP model fine-tuning...")
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        print(f"\tCurrent model name (to save the trained weights as): \t{self.model_code_name}\n")

        def convert_models_to_fp32(model):
            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data.float()
                    p.grad.data = p.grad.data.float()

        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)
        
        writer = SummaryWriter(log_dir=log_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.prompt_path if (self.prompt_path is not None and (self.one_2_one or self.ensemble)) else None
        train_dataset = ImageFolderCustom(train_dir, model_name=self.model_code_name,
                                        max_category_samples=self.upper_category_limit,
                                        prompt_path=prompt_path,
                                        preprocess_fn=self.preprocess, safety=self.safe_load,
                                        img_size=self.preprocess.transforms[0].size,
                                        use_advanced_split=self.advanced_split,  # Enable new split
                                        split_type='train', seed=self.seed,
                                        file_format=self.file_format,
                                        test_ratio=self.test_fraction,
                                        avg=self.avg,
                                        one_2_one=self.one_2_one,
                                        ensemble=self.ensemble)

        train_labels = torch.tensor(train_dataset.targets)

        num_unique_classes = len(set(train_dataset.targets))
        n_classes_for_sampler = min(batch_size, num_unique_classes)

        if self.one_2_one and self.prompt_path is not None:
            print(f"One-to-one prompting enabled. Changing batch size to match number of classes per batch: {n_classes_for_sampler}.")
            batch_size = n_classes_for_sampler # Adjust batch size to match the number of classes sampled per batch for one-to-one prompting
        if n_classes_for_sampler < batch_size:
            print(f"Warning:\tOnly {num_unique_classes} unique classes found in the training dataset.")
            # This warning helps explain why the effective batch size might be smaller than configured.
            print(f"Warning:\tNumber of classes to be sampled per batch is reduced from {batch_size} to {n_classes_for_sampler}.")
        # Pass the capped value to the sampler
        # Since n_samples=1, the effective batch size will now be n_classes_for_sampler.
        train_sampler = CLIP_BalancedBatchSampler(train_labels, n_classes_for_sampler, 1)

        # train_sampler = CLIP_BalancedBatchSampler(train_labels, batch_size, 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                                       collate_fn=collate_with_prompts if self.ensemble else None)
        if self.plot_embeddings:
            print("#####  Plotting prompt embeddings...  ######\n")
            if self.avg:
                self.plot_prompt_embeddings(train_dataloader, batch_idx=0)
            self.plot_image_embeddings_2d(train_dataloader,train_labels, n_batches=100)
            self.plot_image_embeddings_2d(train_dataloader,train_labels, n_batches=100, method="tsne")
        test_dataset = ImageFolderCustom(train_dir, model_name=self.model_code_name,
                                          max_category_samples=self.upper_category_limit,
                                          prompt_path=prompt_path,
                                          preprocess_fn=self.preprocess, safety=self.safe_load,
                                          img_size=self.preprocess.transforms[0].size,
                                          use_advanced_split=self.advanced_split,  # Enable new split
                                          split_type='val', seed=self.seed,
                                          file_format=self.file_format,
                                          test_ratio=self.test_fraction,
                                          avg=self.avg,
                                          one_2_one=self.one_2_one,
                                          ensemble=self.ensemble)
        if self.one_2_one and self.prompt_path is not None:
            test_labels = torch.tensor(test_dataset.targets)
            test_sampler = CLIP_BalancedBatchSampler(test_labels,n_classes_for_sampler, 1)

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_sampler=test_sampler
            )
        else:
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                          collate_fn=collate_with_prompts if self.ensemble else None)

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        num_batches_train = len(train_dataloader)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * num_batches_train,
                                                               eta_min=1e-10)

        print(f"\ttraining batches: \t{num_batches_train}")
        print(f"\tevaluation batches: \t{len(test_dataloader)}")

        out_path = self.checkpoints_dir / f"model_{self.model_code_name.replace('_', '_rev_')}_{num_epochs}e.pt"

        ever_best_accuracy = 0.0

        for epoch in range(num_epochs):
            print(f"\t*\tEpoch: {epoch+1}/{num_epochs}")
            epoch_train_loss, step = 0, 0
            self.model.train()
            for batch in tqdm(train_dataloader, total=num_batches_train, desc="Training"):
                step += 1
                optimizer.zero_grad()

                images, class_ids, prompts = batch
                images = images.to(self.device)

                #print("Batch example:", next(iter(train_dataloader)))
                #exit()

                if self.avg:
                    # Encode images
                    image_features = self.model.encode_image(images)
                    text_features = torch.stack([self.text_features[label_id] for label_id in class_ids]).to(
                        self.device)

                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Calculate logits
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.T.to(image_features.dtype)
                    logits_per_text = logits_per_image.T

                elif self.ensemble:
                    per_image_text_feats = []

                    for prompt_list in prompts:   # prompts är list[list[str]]
                        text_emb = self.aggregate_prompts(prompt_list, strategy="mean")
                        per_image_text_feats.append(text_emb)

                    text_features = torch.stack(per_image_text_feats).to(self.device)

                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.T
                    logits_per_text = logits_per_image.T

                else:

                    if self.one_2_one:
                        # prompts är en lista med metadata‑prompter från datasetet
                        #combined_prompts = [
                        #    f"a scan of {self.texts[label_id]} {prompt}"
                        #    for label_id, prompt in zip(class_ids, prompts)
                        #]
                        combined_prompts = prompts
                        texts = clip.tokenize(combined_prompts).to(self.device)
                    else:
                        # originalbeteende
                        texts = [f"a scan of {self.texts[label_id]}" for label_id in class_ids]
                        texts = clip.tokenize(texts).to(self.device)

                    logits_per_image, logits_per_text = self.model(images, texts)


                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,
                                                                                        ground_truth)) / 2
                total_train_loss.backward()
                epoch_train_loss += total_train_loss.item()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                if self.device != "cpu":
                    convert_models_to_fp32(self.model)
                optimizer.step()
                if self.device != "cpu":
                    clip.model.convert_weights(self.model)
                scheduler.step()

                if step % 25 == 0:
                    print(
                        f"Step {step}/{num_batches_train}, Loss: {total_train_loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")

            avg_train_loss = epoch_train_loss / num_batches_train
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
            print(f"\n{self.model_code_name}\t Epoch {epoch+1} train loss:\t{avg_train_loss:.4f}\n")

            if epoch == num_epochs - 1:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                        'rnd_data_seed': self.seed,
                        'category_limit': self.upper_category_limit,
                    },
                    out_path)
                print(f"Saved weights to {out_path}")

            # Evaluation
            self.model.eval()

            if self.avg:
                num_classes = len(self.categories)  # Use self.categories when avg=True
            else:
                num_classes = len(test_dataset.classes)  # Use dataset classes when avg=False

            acc_top1_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            acc_top3_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3).to(self.device)

            # For evaluation, always use ALL pre-computed text features
            if self.avg:
                text_features_eval = self.text_features  # Use the pre-computed clip set of averaged text features
                text_features_eval = text_features_eval / text_features_eval.norm(dim=-1,
                                                                                  keepdim=True)  # Ensure normalized
            elif self.ensemble:
                text_features_eval = None   # vi beräknar per bild i loopen

            else:
                if self.one_2_one:
                    text_features_eval = None   # vi beräknar textfeatures per batch
                else:
                    all_texts = torch.cat([clip.tokenize(f"a scan of {c}") for c in self.texts]).to(self.device)
                    with torch.no_grad():
                        text_features_eval = self.model.encode_text(all_texts)
                        text_features_eval = text_features_eval / text_features_eval.norm(dim=-1, keepdim=True)


            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating"):
                    images, class_ids, prompts = batch
                    images, class_ids = images.to(self.device), class_ids.to(self.device)

                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    if self.one_2_one:
                        # Encode prompts per batch
                        texts = clip.tokenize(prompts).to(self.device)
                        text_features = self.model.encode_text(texts)
                        text_features_eval = text_features / text_features.norm(dim=-1, keepdim=True)
                        #similarity = (100.0 * (image_features * text_features_eval).sum(dim=-1, keepdim=True))
                    elif self.ensemble:
                        per_image_text_feats = []

                        for prompt_list in prompts:   # prompts är list[list[str]]
                            text_emb = self.aggregate_prompts(prompt_list, strategy="mean")
                            per_image_text_feats.append(text_emb)

                        text_features_eval = torch.stack(per_image_text_feats).to(self.device)
                        text_features_eval = text_features_eval / text_features_eval.norm(dim=-1, keepdim=True)
                    if self.ensemble:
                        # använd klassernas medelvärdes-embeddings för klassificering
                        class_text_feats = self.text_features.to(self.device)
                        class_text_feats = class_text_feats / class_text_feats.norm(dim=-1, keepdim=True)

                        similarity = (100.0 * image_features @ class_text_feats.T)


                    else:
                        similarity = (100.0 * image_features @ text_features_eval.T)
                    #print("\n--- DEBUG SHAPES ---")
                    #print("similarity.shape:", similarity.shape)
                    #print("class_ids.shape:", class_ids.shape)
                    #print("unique class_ids:", class_ids.unique())
                    #print("num_classes (model):", self.num_classes)
                    #print("one2one:", self.one_2_one)
                    #print("---------------------\n")
                    acc_top1_metric.update(similarity, class_ids)
                    acc_top3_metric.update(similarity, class_ids)

            mean_top1_accuracy = acc_top1_metric.compute()
            mean_top3_accuracy = acc_top3_metric.compute()

            if mean_top1_accuracy > ever_best_accuracy:
                ever_best_accuracy = mean_top1_accuracy
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                        'rnd_data_seed': self.seed,
                        'category_limit': self.upper_category_limit,
                    }, out_path)
                print(
                    f"\tSaved checkpoint weights to {out_path}")

            print(f"Mean Top 1 Accuracy: {mean_top1_accuracy.item() * 100:.2f}%.")
            print(f"Mean Top 3 Accuracy: {mean_top3_accuracy.item() * 100:.2f}%.")
            writer.add_scalar("Test Accuracy/Top1", mean_top1_accuracy, epoch)
            writer.add_scalar("Test Accuracy/Top3", mean_top3_accuracy, epoch)

            acc_top1_metric.reset()
            acc_top3_metric.reset()

        writer.flush()
        writer.close()
        print(f"\t*\tFine-tuning of {self.model_code_name} is finished.")

        if not Path(self.models_dir).is_dir():
            os.makedirs(self.models_dir, exist_ok=True)
        self.save_model(str(self.models_dir))

        test_dataset = ImageFolderCustom(train_dir if eval_dir is None else eval_dir,
                                         model_name=self.model_code_name,
                                         max_category_samples=self.upper_category_limit,
                                         prompt_path=prompt_path,
                                         preprocess_fn=self.preprocess,
                                         img_size=self.preprocess.transforms[0].size,
                                         use_advanced_split=(eval_dir is None),  # Enable new split
                                         split_type='test', seed=self.seed,
                                         file_format=self.file_format,
                                         test_ratio=self.test_fraction,
                                        avg=self.avg,
                                          one_2_one=self.one_2_one,
                                          ensemble=self.ensemble)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      collate_fn=collate_with_prompts if self.ensemble else None)

        self.test(test_dataloader, image_files=test_dataset.paths)

    def evaluate_saved_model(self, model_path: str, eval_dir: str, batch_size: int = 8):
        """
        Loads a saved model and evaluates its performance on the specified evaluation directory.
        """
        if model_path is not None:
            if Path(model_path).is_file():
                print(f"Loading model from {model_path} for evaluation...")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.4f}.")
            else:
                print(f"Loading from directory {model_path} using HF Hub mixin...")
                self.load_model(load_directory=model_path, revision=model_path.split("_")[-1] if "_" in model_path else "main")

        model_name = Path(model_path).stem if model_path is not None else self.model_code_name
        prompt_path = self.prompt_path if (self.prompt_path is not None and (self.one_2_one or self.ensemble)) else None
        eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=None,
                                         prompt_path=prompt_path,
                                         preprocess_fn=self.preprocess, img_size=self.preprocess.transforms[0].size,
                                         file_format=self.file_format, use_advanced_split=False,
                                         split_type='test', seed=self.seed, model_name=model_name,
                                         avg=self.avg, one_2_one=self.one_2_one, ensemble=self.ensemble)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

        print("Starting evaluation of the loaded model...")
        self.test(eval_dataloader, image_files=eval_dataset.paths)
        print("Evaluation finished.")

    def top_N_prediction(self, image_data: torch.Tensor, N: int):
        """
        Predicts the top N categories for a given image tensor.
        :param image_data: A tensor of shape (1, 3, H, W) representing the image.
        :param N: The number of top categories to return.
        """
        image_data = image_data.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_data)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            if self.avg or self.zero_shot:
                # Use pre-computed averaged features for efficiency
                text_features = self.text_features
                similarity = (100.0 * image_features @ text_features.T)
                probs = similarity.softmax(dim=-1).cpu().numpy()
            else:
                logits_per_image, _ = self.model(image_data, self.text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        pred_scores = probs[0]
        best_n_indices = np.argsort(pred_scores)[-N:][::-1]
        best_n_scores = pred_scores[best_n_indices]
        return best_n_scores, best_n_indices, pred_scores

    def prediction(self, image_data: torch.Tensor) -> (np.array, int):
        """
        Predicts the top N categories for a single image tensor and returns the scores and indices.
        :param image_data:
        :return:
        """
        # This method is less used, but we'll align it.
        scores, indices, _ = self.top_N_prediction(image_data.unsqueeze(0), len(self.categories))
        return scores, indices[0]

    def test(self, test_dataloader: torch.utils.data.DataLoader, image_files: list,
             vis: bool = True, tab: bool = True):
        """
        Evaluates the model on the provided test dataloader and generates a confusion matrix plot.
        :param test_dataloader:
        :param vis:
        :return:
        """

        plot_path = Path(f'{self.output_dir}/plots')
        table_path = Path(f'{self.output_dir}/tables')
        plot_path.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d-%H%M")

        all_pred_scores = []
        all_predictions = []
        all_true_labels = []

        self.model.eval()

        if self.avg:
            text_features_test = self.text_features  # Use the pre-computed cli set of averaged text features
            text_features_test /= text_features_test.norm(dim=-1, keepdim=True)  # Ensure normalized
        else:
            # Original logic for non-averaged categories
            # The `test_dataloader.dataset.texts` is not directly accessible if not `self.avg`.
            # Instead, use the `self.texts` which holds all descriptions.
            all_texts = torch.cat(
                [clip.tokenize(f"a scan of {c}") for c in self.texts]).to(self.device)
            with torch.no_grad():
                text_features_test = self.model.encode_text(all_texts)
                text_features_test /= text_features_test.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            for images, class_ids, prompts in tqdm(test_dataloader, desc="Testing"):
                images = images.to(self.device)
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                if self.one_2_one:

                    texts = clip.tokenize(prompts).to(self.device)
                    text_features = self.model.encode_text(texts)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    similarity = 100.0 * (image_features * text_features).sum(dim=-1, keepdim=True)

                else:

                    similarity = (100.0 * image_features @ text_features_test.T)

                all_pred_scores.append(similarity.cpu().numpy())

                _, predicted_labels = similarity.max(dim=1)

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_true_labels.extend(class_ids.cpu().numpy())
                # images.append(images.cpu().numpy())

        acc = round(100 * np.sum(np.array(all_predictions) == np.array(all_true_labels)) / len(all_true_labels), 2)
        print("=" * 40)
        print('\t*\tAccuracy: ', acc)
        print("=" * 40)

        all_pred_scores = np.vstack(all_pred_scores)

        # max_logits = np.max(all_pred_scores, axis=1, keepdims=True)
        # exp_logits = np.exp(all_pred_scores - max_logits)
        # all_pred_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        number_of_samples = all_pred_scores.shape[0]

        plot_image = plot_path / f'{time_stamp}_{number_of_samples}{"_zero" if self.zero_shot else ""}_EVAL_TOP-{self.top_N}_{self.model_code_name}.png'
        table_file = table_path / f'{time_stamp}_{number_of_samples}{"_zero" if self.zero_shot else ""}_EVAL_TOP-{self.top_N}_{self.model_code_name}.csv'
        plot_report= plot_path / f'{time_stamp}_{number_of_samples}{"_zero" if self.zero_shot else ""}_CLASSI_REPORT_TOP-{self.top_N}_{self.model_code_name}.png'

        if vis:
            # Ensure display labels match the order of predictions
            display_labels = self.categories if self.avg else test_dataloader.dataset.classes
            num_classes = len(display_labels)
            labels = list(range(num_classes))
            report = classification_report(
                    np.array(all_true_labels),
                    np.array(all_predictions),
                    labels=labels,
                    target_names=display_labels,
                    digits=3,
                    zero_division=0)
            main_class_report(report,plot_report)
            disp = ConfusionMatrixDisplay.from_predictions(
                np.array(all_true_labels), np.array(all_predictions),labels=labels,
                  cmap='inferno',
                #normalize="true",
                  display_labels=np.array(display_labels)
            )
            tick_positions = disp.ax_.get_xticks()
            short_labels = [f"{label[0]}{label.split('_')[-1][0] if '_' in label else ''}" for label in disp.display_labels]
            disp.ax_.set_xticks(tick_positions)
            disp.ax_.set_xticklabels(short_labels)

            disp.ax_.set_title(f"TOP {self.top_N} {self.model_code_name} CM  - {acc}%")
            plt.savefig(plot_image, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Confusion matrix saved to {plot_image}")

        if tab:
            out_df, _ = dataframe_results(test_images= image_files, test_predictions=all_pred_scores, raw_scores=None,
                                          top_N= self.top_N, categories=self.categories)
            all_true_labels = np.asarray(all_true_labels, dtype=int)
            out_df["TRUE"] = [self.categories[i] for i in all_true_labels]
            out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
            out_df.to_csv(table_file, sep=",", index=False)
            print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory:\n{table_file}")

        return acc


    def save_model(self, save_directory: str):
        """
        Save the fine-tuned model and processor to the specified directory using PyTorchModelHubMixin.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # The PyTorchModelHubMixin's save_pretrained method handles saving the model and its configuration.
        configs = {
            "vision_feat_dim": self.preprocess.transforms[0].size,
            "text_feat_dim": self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512, # Assuming text feature dim from CLIP
            "model_name": self.model_name,
            "avg": self.avg,
            "categories": self.categories,
            "texts": self.texts, # Save the texts for reconstruction
            "zero_shot": self.zero_shot,
            "rnd_data_seed": self.seed,
            "image_size": self.preprocess.transforms[0].size,
            "image_mean": self.preprocess.transforms[-1].mean,
            "image_std": self.preprocess.transforms[-1].std
        }

        # expand config with base model configs
        if hasattr(self.model, 'config'):
            configs.update(self.model.config.to_dict())
            configs.update(self.preprocess.config.to_dict() if hasattr(self.preprocess, 'config') else {})

        self.save_pretrained(save_directory, config=configs)
        print(f"Model and configuration saved to {save_directory}")

    def load_model(self, load_directory: str, revision: str):
        """
        Load a fine-tuned model and its configuration from the specified directory using PyTorchModelHubMixin.
        """
        # The from_pretrained method of PyTorchModelHubMixin loads the model into the current instance.
        # It also handles loading the associated configuration.
        loaded_model = self.from_pretrained(load_directory,
                                            avg=self.avg,
                                            revision=revision,
                                            model_revision=revision,
                                            seed=self.seed,
                                            max_category_samples=self.upper_category_limit,
                                            eval_max_category_samples=self.upper_category_limit_eval,
                                            top_N=self.top_N,
                                            model_name=self.model_name,  # will be set from config if needed
                                            device=self.device,
                                            categories_tsv=self.categories_tsv,
                                            categories_dir=self.categories_dir
                                            )
        self.model = loaded_model.model
        self.preprocess = loaded_model.preprocess # Assuming preprocess is also part of the loaded state or can be re-initialized

        # Re-initialize other necessary attributes from the loaded configuration
        self.model_name = loaded_model.model_name if hasattr(loaded_model, 'model_name') else self.model_name
        self.avg = loaded_model.avg if hasattr(loaded_model, 'avg') else self.avg
        self.categories = loaded_model.categories if hasattr(loaded_model, 'categories') else self.categories
        self.texts = loaded_model.texts if hasattr(loaded_model, 'texts') else self.texts
        self.zero_shot = loaded_model.zero_shot if hasattr(loaded_model, 'zero_shot') else self.zero_shot


        # Recompute text features if `avg` is true and `texts` were loaded
        if self.avg and hasattr(self, 'texts') and self.texts:
            print("Recomputing averaged text features after loading model.")
            self.text_features = self._get_averaged_text_features()

        self.num_prediction_classes = len(self.categories)
        print(f"Model and configuration loaded from {load_directory}")


    def pushing_to_hub(self, repo_id: str, private: bool = False,
                    token: str = None, revision: str = "main"):
        """
        Upload the fine-tuned model and its configuration to the Hugging Face Model Hub.

        Args:
            repo_id (str): The name of the repository to create or update on the Hugging Face Hub (e.g., "username/my-clip-model").
            private (bool, optional): Whether the repository should be private. Defaults to False.
            token (str, optional): The authentication token for Hugging Face Hub. Defaults to None.
            revision (str, optional): The revision (branch) to push to. Defaults to "main".
        """
        # The PyTorchModelHubMixin's push_to_hub method handles saving the model locally
        # and then pushing it to the Hugging Face Hub.
        # Ensure the config includes necessary parameters for re-instantiation.
        self.push_to_hub(repo_id, private=private, token=token, branch=revision, config={
            "vision_feat_dim": self.preprocess.transforms[0].size,
            "text_feat_dim": self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512,
            "model_name": self.model_name,
            "avg": self.avg,
            "categories": self.categories,
            "texts": self.texts,
            "zero_shot": self.zero_shot,
            "rnf_data_seed": self.seed,
            "image_size": self.preprocess.transforms[0].size,
            "image_mean": self.preprocess.transforms[-1].mean,
            "image_std": self.preprocess.transforms[-1].std
        })
        print(f"Model and configuration pushed to the Hugging Face Hub: {repo_id}")

    def load_from_hub(self, repo_id: str, revision: str = "main"):
        """
        Load a model and its configuration from the Hugging Face Hub.

        Args:
            repo_id (str): The name of the repository on the Hugging Face Hub.
            revision (str, optional): The revision of the repository to load. Defaults to "main".
        """
        # The from_pretrained method of PyTorchModelHubMixin loads the model and its configuration
        # directly into the current instance.
        loaded_model = self.from_pretrained(repo_id, revision=revision,
                                            avg=self.avg, seed=self.seed,
                                            model_revision=revision,
                                            max_category_samples=self.upper_category_limit,
                                            eval_max_category_samples=self.upper_category_limit_eval,
                                            top_N=self.top_N,
                                            model_name=self.model_name,  # will be set from config if needed
                                            device=self.device,
                                            categories_tsv=self.categories_tsv,
                                            categories_dir=self.categories_dir
                                            )

        self.model = loaded_model.model
        self.preprocess = loaded_model.preprocess # Assuming preprocess is also part of the loaded state or can be re-initialized

        # Re-initialize other necessary attributes from the loaded configuration
        self.model_name = loaded_model.model_name if hasattr(loaded_model, 'model_name') else self.model_name
        self.avg = loaded_model.avg if hasattr(loaded_model, 'avg') else self.avg
        self.categories = loaded_model.categories if hasattr(loaded_model, 'categories') else self.categories
        self.texts = loaded_model.texts if hasattr(loaded_model, 'texts') else self.texts
        self.zero_shot = loaded_model.zero_shot if hasattr(loaded_model, 'zero_shot') else self.zero_shot

        # Recompute text features if `avg` is true and `texts` were loaded
        if self.avg and hasattr(self, 'texts') and self.texts:
            print("Recomputing averaged text features after loading model from Hub.")
            self.text_features = self._get_averaged_text_features()

        self.num_prediction_classes = len(self.categories)
        print(f"Model and configuration loaded from the Hugging Face Hub: {repo_id}")


    def predict_single_best(self, image_file: str) -> dict:
        """
        Predicts the category of a single image file.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        best_n_scores, best_n_indices, _ = self.top_N_prediction(image_input, self.top_N)
        pred_label = self.categories[best_n_indices[0]]
        return pred_label
        # results = {label : score for label, score in zip([self.categories[i] for i in best_n_indices], best_n_scores)}
        # return results


    def predict_top_N(self, image_file: str) -> (list, list):
        """
        Predicts the TOP-N categories of a single image file.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        best_n_scores, best_n_indices, _ = self.top_N_prediction(image_input, self.top_N)
        best_n_scores = np.round(best_n_scores, 3).tolist()
        pred_labels = [self.categories[i] for i in best_n_indices]
        return best_n_scores, pred_labels

    def predict_directory(self, folder_path: str, raw: bool = False, out_table: str = None,
                          chunk_size: int = 1000):
        """
        Predicts categories for all images in a directory and saves results to a CSV file.
        Handles large directories (30,000+ files) efficiently with batch processing.

        :param folder_path: Path to directory containing images
        :param raw: Whether to save raw scores
        :param out_table: Optional custom output path for results
        :param batch_size: Number of images to process before writing to disk
        :param recursive: Whether to search subdirectories
        :return:
        """
        folder_path = Path(folder_path)

        images = directory_scraper(Path(folder_path), self.file_format)
        print(f"Found {len(images)} images in {folder_path}")

        time_stamp = time.strftime("%Y%m%d-%H%M")

        # Prepare output paths
        out_table = out_table if out_table is not None \
            else f"{self.output_dir}/tables/{time_stamp}_{len(images)}_{'zero_' if self.zero_shot else ''}result_{self.model_code_name}_TOP-{self.top_N}.csv"

        raw_table = f"{self.output_dir}/tables/{time_stamp}_{len(images)}_{'zero_' if self.zero_shot else ''}RAW_{self.model_code_name}.csv" if raw else None

        # Process in batches
        total_processed = 0
        write_header = True

        for batch_start in range(0, len(images), chunk_size):
            batch_end = min(batch_start + chunk_size, len(images))
            batch_images = images[batch_start:batch_end]

            # --- FIX: Renamed lists for clarity ---
            all_scores_list, tru_images = [], []

            # Process batch
            for img_path in tqdm(batch_images,
                                 desc=f"Processing batch {batch_start // chunk_size + 1}/{(len(images) - 1) // chunk_size + 1}"):
                try:
                    image = Image.open(img_path)
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    scores, indices, raw_scores = self.top_N_prediction(image_input, self.top_N)

                    # --- FIX: Always append the full raw_scores list. Remove res_list (indices). ---
                    all_scores_list.append(raw_scores.tolist())
                    tru_images.append(img_path.name)

                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")
                    continue

            # --- FIX: Check the correct list ---
            if not all_scores_list:
                continue

            # --- FIX: Remove unnecessary concatenation of res_list ---
            # res_list = np.concatenate(res_list, axis=0) # <--- REMOVED

            # --- FIX: Pass the correct lists to dataframe_results ---
            out_df, raw_df = dataframe_results(
                test_images=tru_images,
                test_predictions=all_scores_list,  # <--- Pass the full scores here
                raw_scores=all_scores_list if raw else None, # <--- Pass scores if raw=True, else None
                top_N=self.top_N,
                categories=self.categories
            )

            out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)

            # Append to CSV (write header only once)
            out_df.to_csv(out_table, sep=",", index=False,
                          mode='w' if write_header else 'a',
                          header=write_header)

            if raw:
                raw_df.sort_values(self.categories, ascending=[False] * len(self.categories), inplace=True)
                raw_df.to_csv(raw_table, sep=",", index=False,
                              mode='w' if write_header else 'a',
                              header=write_header)

            write_header = False
            total_processed += len(tru_images)

            # Free memory
            # --- FIX: Update del statement ---
            del all_scores_list, tru_images, out_df
            if raw:
                del raw_df

        print(f"Results for TOP-{self.top_N} predictions ({total_processed} images) saved to {out_table}")
        if raw:
            print(f"RAW Results saved to {raw_table}")

        output_dataframe = pd.read_csv(out_table, sep=",", index_col=None)
        return output_dataframe



    def predict_dir(self, folder_path: str, raw: bool = False, out_table: str = None):
        """
        Predicts categories for all images in a directory and saves results to a CSV file.
        :param folder_path:
        :param raw:
        :param out_table:
        :return:
        """
        images = directory_scraper(Path(folder_path), self.file_format)
        print(f"Predicting {len(images)} images from {folder_path}")

        time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

        res_list, raw_list, tru_images = [], [], []

        for img_path in tqdm(images, desc="Predicting directory"):
            try:
                image = Image.open(img_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                scores, indices, raw_scores = self.top_N_prediction(image_input, self.top_N)
                res_list.append(indices)
                if raw:
                    raw_list.append(raw_scores.tolist())

                tru_images.append(img_path.name)

            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

        res_list = np.concatenate(res_list, axis=0)
        # print(res_list)
        out_df, raw_df = dataframe_results(test_images=tru_images, test_predictions=res_list, raw_scores=raw_list,
                                           top_N=self.top_N, categories=self.categories)

        out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)

        out_table = out_table if out_table is not None \
            else f"{self.output_dir}/tables/{time_stamp}_result_{self.model_code_name}_TOP-{self.top_N}.csv"
        out_df.to_csv(out_table, sep=",", index=False)
        print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory")

        if raw:
            raw_df.sort_values(self.categories, ascending=[False] * len(self.categories), inplace=True)
            raw_df.to_csv(f"{self.output_dir}/tables/{time_stamp}_RAW_{self.model_code_name}.csv", sep=",", index=False)
            print(f"RAW Results are recorded into {self.output_dir}/tables/ directory")

    def predict_directory_chunk(self, folder_path: str, raw: bool = False, out_table: str = None, chunk: int = 1000):
        """
        Predicts categories for all images in a directory and saves results to a CSV file.
        If chunk > 0, results are saved every `chunk` images into the same CSV (append mode).
        :param folder_path:
        :param raw:
        :param out_table:
        :param chunk: number of images to process before flushing results to disk (0 or None => no chunking)
        :return:
        """

        images = directory_scraper(Path(folder_path), self.file_format)
        n_images = len(images)
        print(f"Predicting {n_images} images from {folder_path}")

        time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

        # If user didn't pass an explicit out_table use default naming
        out_table = out_table if out_table is not None \
            else f"{self.output_dir}/tables/{time_stamp}_result_{self.model_code_name}_TOP-{self.top_N}.csv"

        # Raw results file name (same pattern as before)
        raw_out_table = f"{self.output_dir}/tables/{time_stamp}_RAW_{self.model_code_name}.csv"

        # For non-chunking behavior we keep similar buffers to your original code
        if not chunk:
            res_list, raw_list, tru_images = [], [], []
            for img_path in tqdm(images, desc="Predicting directory"):
                try:
                    image = Image.open(img_path)
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    scores, indices, raw_scores = self.top_N_prediction(image_input, self.top_N)
                    res_list.append(indices)
                    if raw:
                        raw_list.append(raw_scores.tolist())
                    tru_images.append(img_path.name)
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")

            if res_list:
                res_array = np.concatenate(res_list, axis=0)
            else:
                res_array = np.empty((0, self.top_N), dtype=int)

            out_df, raw_df = dataframe_results(test_images=tru_images, test_predictions=res_array,
                                               raw_scores=raw_list, top_N=self.top_N, categories=self.categories)

            out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)

            # ensure output directory exists
            os.makedirs(os.path.dirname(out_table), exist_ok=True)
            out_df.to_csv(out_table, sep=",", index=False)
            print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory")

            if raw:
                raw_df.sort_values(self.categories, ascending=[False] * len(self.categories), inplace=True)
                raw_df.to_csv(raw_out_table, sep=",", index=False)
                print(f"RAW Results are recorded into {self.output_dir}/tables/ directory")

            return

        # --- Chunking / incremental write mode ---
        # chunk > 0 here
        os.makedirs(os.path.dirname(out_table), exist_ok=True)

        chunk_images, chunk_res_list, chunk_raw_list = [], [], []
        wrote_header = False
        processed = 0

        for img_path in tqdm(images, desc="Predicting directory"):
            try:
                image = Image.open(img_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                scores, indices, raw_scores = self.top_N_prediction(image_input, self.top_N)

                chunk_images.append(img_path.name)
                chunk_res_list.append(indices)  # indices expected to be something concatenable along axis=0
                if raw:
                    chunk_raw_list.append(raw_scores.tolist())

                processed += 1

                # flush when we have chunk images (or on last iteration below)
                if processed % chunk == 0:
                    # prepare arrays/lists for dataframe_results
                    if chunk_res_list:
                        chunk_res_array = np.concatenate(chunk_res_list, axis=0)
                    else:
                        chunk_res_array = np.empty((0, self.top_N), dtype=int)

                    out_df_chunk, raw_df_chunk = dataframe_results(test_images=chunk_images,
                                                                   test_predictions=chunk_res_array,
                                                                   raw_scores=chunk_raw_list,
                                                                   top_N=self.top_N, categories=self.categories)

                    # append chunk to CSV
                    mode = 'w' if not wrote_header else 'a'
                    header = not wrote_header
                    out_df_chunk.to_csv(out_table, sep=",", index=False, mode=mode, header=header)
                    wrote_header = True
                    print(f"Flushed {processed} images -> {out_table}")

                    if raw:
                        # append raw chunk to raw_out_table (sort later)
                        mode_raw = 'w' if not os.path.exists(raw_out_table) else 'a'
                        header_raw = not os.path.exists(raw_out_table)
                        raw_df_chunk.to_csv(raw_out_table, sep=",", index=False, mode=mode_raw, header=header_raw)

                    # reset chunk buffers
                    chunk_images, chunk_res_list, chunk_raw_list = [], [], []

            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

        # flush remaining images (if any)
        if chunk_images:
            if chunk_res_list:
                chunk_res_array = np.concatenate(chunk_res_list, axis=0)
            else:
                chunk_res_array = np.empty((0, self.top_N), dtype=int)

            out_df_chunk, raw_df_chunk = dataframe_results(test_images=chunk_images,
                                                           test_predictions=chunk_res_array,
                                                           raw_scores=chunk_raw_list,
                                                           top_N=self.top_N, categories=self.categories)

            mode = 'w' if not wrote_header else 'a'
            header = not wrote_header
            out_df_chunk.to_csv(out_table, sep=",", index=False, mode=mode, header=header)
            print(f"Flushed final {processed} images -> {out_table}")

            if raw:
                mode_raw = 'w' if not os.path.exists(raw_out_table) else 'a'
                header_raw = not os.path.exists(raw_out_table)
                raw_df_chunk.to_csv(raw_out_table, sep=",", index=False, mode=mode_raw, header=header_raw)

        # Final pass: read entire CSV, sort, and re-write so final file is sorted like original behavior
        try:
            final_out = pd.read_csv(out_table)
            if {'FILE', 'PAGE'}.issubset(final_out.columns):
                final_out.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
            final_out.to_csv(out_table, sep=",", index=False)
            print(f"Final sorted results written to {out_table}")
        except Exception as e:
            print(f"Warning: could not reload/sort final output file {out_table}: {e}")

        if raw:
            try:
                final_raw = pd.read_csv(raw_out_table)
                # original behavior sorts raw_df by categories descending
                if all(c in final_raw.columns for c in self.categories):
                    final_raw.sort_values(self.categories, ascending=[False] * len(self.categories), inplace=True)
                final_raw.to_csv(raw_out_table, sep=",", index=False)
                print(f"Final RAW results written to {raw_out_table}")
            except Exception as e:
                print(f"Warning: could not reload/sort final RAW file {raw_out_table}: {e}")

    def load_dataset(self, train_dir: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Loads the dataset from the specified directory and returns a DataLoader with a balanced batch sampler.
        The dataset is expected to be organized in subdirectories for each class, compatible with ImageFolder"""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.prompt_path if (self.prompt_path is not None and (self.one_2_one or self.ensemble)) else None
        train_dataset = ImageFolderCustom(train_dir, model_name=self.model_code_name,
                                        max_category_samples=self.upper_category_limit,
                                        prompt_path=prompt_path,
                                        preprocess_fn=self.preprocess, safety=self.safe_load,
                                        img_size=self.preprocess.transforms[0].size,
                                        use_advanced_split=self.advanced_split,  # Enable new split
                                        split_type='test', seed=self.seed,
                                        file_format=self.file_format,
                                        test_ratio=self.test_fraction,
                                        avg=self.avg,
                                        one_2_one=self.one_2_one,
                                        ensemble=self.ensemble)

        train_labels = torch.tensor(train_dataset.targets)


        num_unique_classes = len(set(train_dataset.targets))
        n_classes_for_sampler = min(batch_size, num_unique_classes)

        if self.one_2_one and self.prompt_path is not None:
            print(f"One-to-one prompting enabled. Changing batch size to match number of classes per batch: {n_classes_for_sampler}.")
            batch_size = n_classes_for_sampler # Adjust batch size to match the number of classes sampled per batch for one-to-one prompting
        if n_classes_for_sampler < batch_size:
            print(f"Warning:\tOnly {num_unique_classes} unique classes found in the training dataset.")
            # This warning helps explain why the effective batch size might be smaller than configured.
            print(f"Warning:\tNumber of classes to be sampled per batch is reduced from {batch_size} to {n_classes_for_sampler}.")
        # Pass the capped value to the sampler
        # Since n_samples=1, the effective batch size will now be n_classes_for_sampler.
        train_sampler = CLIP_BalancedBatchSampler(train_labels, n_classes_for_sampler, 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                                    collate_fn=collate_with_prompts if self.ensemble else None)
        
        return train_dataset,train_dataloader

    def plot_prompt_embeddings(self, dataloader, batch_idx=0, max_chars=300):
        """
        Plots CLIP text embeddings for prompts in a single batch.
        Similar to _get_averaged_text_features(), but for per-image prompts.
        """

        import matplotlib.pyplot as plt
        import os
        from itertools import islice

        # Hämta en batch
        batch = next(islice(dataloader, batch_idx, None))
        images, class_ids, prompts = batch

        # Trunka långa prompts (CLIP max 77 tokens)
        prompts = [p[:max_chars] for p in prompts]

        # Tokenisera
        tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Medelvärde
        mean_features = text_features.mean(dim=0)
        mean_features = mean_features / mean_features.norm()

        # --- PLOTTING ---
        num_prompts = text_features.shape[0]
        total_plots = num_prompts + 1

        fig, axes = plt.subplots(total_plots, 1, figsize=(10, 1.2 * total_plots), sharex=True)

        for i in range(num_prompts):
            axes[i].imshow(text_features[i].unsqueeze(0).cpu().numpy(),
                        aspect="auto", cmap="viridis")
            axes[i].set_ylabel(f"p{i}", rotation=0, labelpad=20)
            axes[i].set_yticks([])

        axes[-1].imshow(mean_features.unsqueeze(0).cpu().numpy(),
                        aspect="auto", cmap="viridis")
        axes[-1].set_ylabel("mean", rotation=0, labelpad=20)
        axes[-1].set_yticks([])

        plt.suptitle("Prompt embedding dimensions (batch {})".format(batch_idx))
        plt.xlabel("embedding dimension")
        plt.tight_layout()

        outdir = self.output_dir / "prompt_embeddings"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outdir / f"{self.model_code_name}_batch{batch_idx}_prompt_embeddings.png")

        print(f"Saved prompt embedding plot for batch {batch_idx} → {outdir}")

    def plot_image_embeddings_2d_old(self, dataloader, labels, n_batches=100, method="pca"):
        """
        Collects embeddings from multiple batches and plots them in 2D.
        """

        #import matplotlib.pyplot as plt
        #import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        #import numpy as np
        #import os

        all_feats = []
        all_classes = []

        # --- 1. Hämta flera batchar ---
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            images, class_ids, prompts = batch

            with torch.no_grad():
                feats = self.model.encode_image(images.to(self.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy())
                all_classes.append(class_ids.cpu().numpy())

        # --- 2. Slå ihop allt ---
        all_feats = np.concatenate(all_feats, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)

        # --- 3. Reducera dimensioner ---
        if method.lower() == "pca":
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(all_feats)

        elif method.lower() == "tsne":
            # auto-perplexity baserat på antal samples
            n_samples = all_feats.shape[0]
            perp = max(2, min(30, n_samples // 3))
            reducer = TSNE(
                n_components=2,
                perplexity=perp,
                learning_rate="auto",
                init="random"
            )
            reduced = reducer.fit_transform(all_feats)

        else:
            raise ValueError("method must be 'pca' or 'tsne'")

        # --- 4. Plot ---
        outdir = self.output_dir / "image_embeddings_2d"
        os.makedirs(outdir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced[:, 0],
            y=reduced[:, 1],
            hue=all_classes,
            palette="tab10",
            s=50,
            edgecolor="black"
        )

        plt.title(f"2D Image Embeddings ({method.upper()}) — {n_batches} batches")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Class ID", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        outfile = outdir / f"{self.model_code_name}_{method}_{n_batches}batches.png"
        plt.savefig(outfile)

        print(f"Saved 2D image embedding plot → {outfile}")

    def plot_image_embeddings_2d(self, dataloader, labels=None, n_batches=100, method="pca"):
        """
        Collect embeddings from multiple batches and produce two 2D plots:
        - image embeddings (image_feats)
        - text embeddings (text_feats from prompts)
        Saves PNGs to self.output_dir / "image_embeddings_2d".
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import torch

        device = getattr(self, "device", "cpu")


        plot_path = Path(f'{self.output_dir}/plots')
        table_path = Path(f'{self.output_dir}/tables')
        plot_path.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d-%H%M")


        all_image_feats = []
        all_text_feats = []
        all_classes = []
        sample_prompts = []

        # --- 1. Hämta flera batchar ---
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            # Antag batch = (images, class_ids, prompts)
            images, class_ids, prompts = batch

            # Image embeddings
            with torch.no_grad():
                img_feats = self.model.encode_image(images.to(device))
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                all_image_feats.append(img_feats.cpu().numpy())

            # Text embeddings: prompts kan vara list/tuple av strings
            # Tokenisera/encode per batch; om prompts redan tokeniserade, anpassa här
            try:
                # clip.tokenize if you used raw text; here we assume model.encode_text accepts tokenized input
                # If your model.encode_text expects token ids, replace below with appropriate tokenization.
                text_tokens = clip.tokenize(list(prompts)).to(device)
                with torch.no_grad():
                    txt_feats = self.model.encode_text(text_tokens)
                    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                    all_text_feats.append(txt_feats.cpu().numpy())
            except Exception:
                # Fallback: if prompts are already token tensors or encode_text accepts raw strings
                try:
                    with torch.no_grad():
                        txt_feats = self.model.encode_text(prompts)  # if encode_text handles list[str]
                        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                        all_text_feats.append(txt_feats.cpu().numpy())
                except Exception:
                    # If text encoding fails, skip this batch's text embeddings but keep images
                    all_text_feats.append(np.zeros((len(prompts), self.model.visual.output_dim)))  # placeholder

            all_classes.append(class_ids.cpu().numpy())
            sample_prompts.extend(list(prompts))

        if len(all_image_feats) == 0:
            print("No image features collected.")
            return

        # --- 2. Slå ihop allt ---
        all_image_feats = np.concatenate(all_image_feats, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        try:
            all_text_feats = np.concatenate(all_text_feats, axis=0)
        except Exception:
            all_text_feats = None

        outdir = Path(self.output_dir) / "image_embeddings_2d"
        os.makedirs(outdir, exist_ok=True)

        def reduce_and_plot(feats, classes, title_suffix, fname_suffix):
            n_samples = feats.shape[0]
            if n_samples < 2:
                print(f"Not enough samples for {title_suffix} (n={n_samples}). Skipping.")
                return

            if method.lower() == "pca" or n_samples < 10:
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(feats)
            elif method.lower() == "tsne":
                perp = max(2, min(30, max(2, n_samples // 3)))
                reducer = TSNE(n_components=2, perplexity=perp, learning_rate="auto", init="random")
                reduced = reducer.fit_transform(feats)
            else:
                raise ValueError("method must be 'pca' or 'tsne'")

            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=classes, palette="tab10", s=50, edgecolor="black", legend="full")
            plt.title(f"{title_suffix} ({method.upper()}) — {n_samples} samples")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.legend(title="Class ID", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            outfile = outdir / f"{self.model_code_name}_{fname_suffix}_{method}_{n_batches}batches.png"
            plt.savefig(outfile)
            plt.close()
            print(f"Saved {title_suffix} plot → {outfile}")

        # --- 3. Plot image embeddings ---
        reduce_and_plot(all_image_feats, all_classes, "Image Embeddings", "image_embeddings")

        # --- 4. Plot text embeddings (if available) ---
        if all_text_feats is not None and all_text_feats.shape[0] == all_image_feats.shape[0]:
            reduce_and_plot(all_text_feats, all_classes, "Text (Prompt) Embeddings", "text_embeddings")
        elif all_text_feats is not None:
            # If text and image counts differ, plot text separately with its own class mapping if possible
            text_classes = all_classes[: all_text_feats.shape[0]] if all_text_feats.shape[0] <= all_classes.shape[0] else np.arange(all_text_feats.shape[0])
            reduce_and_plot(all_text_feats, text_classes, "Text (Prompt) Embeddings", "text_embeddings")
        else:
            print("No text embeddings available to plot.")

    def build_image_index(self, dataloader):
        """Extract image features for full dataset and store them for fast retrieval. Also keep track of image IDs."""
        self.all_image_feats = []
        self.all_image_ids = []
        self.model.eval()
        with torch.no_grad():
            for images, class_ids, meta in dataloader:
                feats = self.model.encode_image(images.to(self.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self.all_image_feats.append(feats.cpu())
                # anta meta innehåller filename eller id
                self.all_image_ids.extend([m if isinstance(m, str) else m['local_filename'] for m in meta])
        self.all_image_feats = torch.cat(self.all_image_feats, dim=0)  # (N, D)

    def search_by_image(self, query_image, k=10):
        """Return top-k most similar images from the indexed dataset given a query image."""
        self.model.eval()
        with torch.no_grad():
            q = self.model.encode_image(self.preprocess(query_image).unsqueeze(0).to(self.device))
            q = q / q.norm(dim=-1, keepdim=True)
            sims = (q @ self.all_image_feats.T).squeeze(0)  # (N,)
            topk = sims.topk(k)
            return [(self.all_image_ids[i], float(sims[i])) for i in topk.indices.tolist()]

    def search_by_text(self, prompt, k=10):
        """Tokenisera prompt, enkoda och returnera top-k bilder."""
        self.model.eval()
        with torch.no_grad():
            tokens = clip.tokenize([prompt]).to(self.device)
            t = self.model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
            sims = (t @ self.all_image_feats.T).squeeze(0)
            topk = sims.topk(k)
            return [(self.all_image_ids[i], float(sims[i])) for i in topk.indices.tolist()]

    def build_text_index(self, prompts_list):
        """Skapa och spara text‑embeddings för alla prompts (lista av str)."""
        self.all_prompts = prompts_list
        self.all_text_feats = []
        self.model.eval()
        batch = []
        B = 128
        for i, p in enumerate(prompts_list):
            batch.append(p)
            if len(batch) == B or i == len(prompts_list)-1:
                tokens = clip.tokenize(batch).to(self.device)
                with torch.no_grad():
                    tf = self.model.encode_text(tokens)
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                    self.all_text_feats.append(tf.cpu())
                batch = []
        self.all_text_feats = torch.cat(self.all_text_feats, dim=0)  # (M, D)

    def retrieve_prompts_for_image(self, image, k=5):
        """Givet en bild, returnera topp‑k prompts (text) som matchar bäst."""
        self.model.eval()
        with torch.no_grad():
            q = self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))
            q = q / q.norm(dim=-1, keepdim=True)
            sims = (q @ self.all_text_feats.T).squeeze(0)
            topk = sims.topk(k)
            return [(self.all_prompts[i], float(sims[i])) for i in topk.indices.tolist()]

    def cluster_dataset(self,dataset, dataloader:torch.utils.data.DataLoader
                        , n_clusters=20, reduce_dim=50,cluster_by_num_classes=True,
                          csv_out="cluster_results.csv"):
        """
        Extract image features, reduce dimension with PCA, cluster with KMeans,
        plot the reduced features, and save results to CSV including categories.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import torch

        if cluster_by_num_classes:
            n_clusters = len(self.categories)

        feats = []
        filenames = []
        categories = []

        plot_path = Path(f'{self.output_dir}/plots')
        table_path = Path(f'{self.output_dir}/tables')
        plot_path.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d-%H%M")

        print("Images loaded for clustering:", len(dataset))

        print("********* Clustering dataset... *********\n")
        print(f"Number of clustering classes: {n_clusters}")
        print("\nImages loaded for clustering:", len(dataset))
        self.model.eval()

        all_paths = dataset.paths
        global_idx = 0

        with torch.no_grad():
            for images, class_ids, meta in tqdm(dataloader, desc="Extracting CLIP features"):
                f = self.model.encode_image(images.to(self.device))
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())

                batch_size = images.size(0)
                batch_paths = all_paths[global_idx : global_idx + batch_size]
                filenames.extend(batch_paths)
                global_idx += batch_size

                for cid in class_ids:
                    categories.append(self.categories[int(cid)])



        feats = np.concatenate(feats, axis=0)

        pca = PCA(n_components=reduce_dim)
        reduced = pca.fit_transform(feats)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced)

        if reduce_dim >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=12)
            plt.title(f"PCA (2D) – {n_clusters} clusters")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.tight_layout()
            #plt.show()

        plot_image = plot_path / f'{time_stamp}_{"_zero" if self.zero_shot else ""}_CLUSTER_PLOT_TOP-{self.top_N}_{self.model_code_name}.png'
        table_file = table_path / f'{time_stamp}_{"_zero" if self.zero_shot else ""}_CLUSTER_TABLE_TOP-{self.top_N}_{self.model_code_name}.csv'
        #plot_report= plot_path / f'{time_stamp}_{"_zero" if self.zero_shot else ""}_CLUSTER_REPORT_TOP-{self.top_N}_{self.model_code_name}.png'
        plt.savefig(plot_image)
        print(f"Saved cluster plot to: {plot_image}")
        df = pd.DataFrame({
            "filename": filenames,
            "cluster": labels,
            "category": categories
        })

        # add PCA coordinates
        for i in range(reduce_dim):
            df[f"pc{i+1}"] = reduced[:, i]

        df.to_csv(table_file, index=False)
        print(f"Saved clustering results to: {table_file}")

        # -----------------------------
        # 6. Return mapping: filename → (cluster, category)
        # -----------------------------
        mapping = {
            fn: {"cluster": cl, "category": cat}
            for fn, cl, cat in zip(filenames, labels, categories)
        }
        if cluster_by_num_classes:
            df_compare = pd.DataFrame({
                    "filename": filenames,
                    "true_label": categories,
                    "cluster": labels
                })

            print("\nCluster vs True Label:")
            print(pd.crosstab(df_compare["true_label"], df_compare["cluster"]))


        return mapping, reduced, labels

def split_data_80_10_10(files: list, labels: list, random_seed: int, max_categ: int,
                        safe_check: bool = True):
    """
    Splits the data into training, validation, and test sets with an 80/10/10 ratio.
    The split uses uniform distribution selection to maintain temporal distribution
    across the sorted files (by creation date). Test and dev sets are selected first,
    with remaining samples going to training.

    Args:
        files: List of file paths (should be sorted alphabetically by creation date)
        labels: List of corresponding labels
        random_seed: Random seed for reproducibility
        max_categ: Maximum number of samples per category to consider
        safe_check: If True, checks for corrupted images and excludes them
    Returns:
        tuple: (train_files, val_files, test_files, train_labels, val_labels, test_labels)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    files = np.array(files)
    labels = np.array(labels)

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        n_samples = len(indices)

        if n_samples > max_categ:
            print(f"Label {label} has {n_samples} samples, limiting to {max_categ}.")
            indices = np.random.choice(indices, size=max_categ, replace=False)

        label_to_indices[label] = indices.tolist()

    total_files = [files[idx] for label in label_to_indices for idx in label_to_indices[label]]
    total_labels = [labels[idx] for label in label_to_indices for idx in label_to_indices[label]]

    if safe_check:
        print(f"Checking {len(total_files)} files for corrupted images...")
        good_files, good_labels = [], []
        for file, label in zip(total_files, total_labels):
            try:
                Image.open(file).load()
                good_files.append(file)
                good_labels.append(label)
            except Exception as e:
                print(f"File {file} is corrupted: {e}")
                continue
        print(f"Total usable images found: {len(good_files)} / {len(total_files)}")
    else:
        good_files, good_labels = total_files, total_labels

    files, labels = np.array(good_files), np.array(good_labels)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    test_indices = []
    val_indices = []
    train_indices = []

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        n_samples = len(indices)

        n_test = max(1, int(n_samples * 0.1))
        n_val = max(1, int(n_samples * 0.1))

        if n_test + n_val > n_samples:
            n_test = n_samples // 2
            n_val = n_samples - n_test

        if n_test > 0:
            test_step = n_samples / n_test
            test_positions = np.arange(0, n_samples, test_step)[:n_test]
            test_positions += np.random.uniform(-test_step / 4, test_step / 4, size=len(test_positions))
            test_positions = np.clip(test_positions, 0, n_samples - 1).astype(int)
            selected_test = indices[test_positions]
            test_indices.extend(selected_test)

        remaining_mask = np.ones(n_samples, dtype=bool)
        if n_test > 0:
            remaining_mask[test_positions] = False
        remaining_indices = indices[remaining_mask]
        n_remaining = len(remaining_indices)

        if n_val > 0 and n_remaining > 0:
            val_step = n_remaining / n_val if n_val <= n_remaining else 1
            val_positions = np.arange(0, n_remaining, val_step)[:n_val]
            if len(val_positions) > n_remaining:
                val_positions = np.arange(n_remaining)
            val_positions += np.random.uniform(-val_step / 4 if val_step > 1 else 0,
                                               val_step / 4 if val_step > 1 else 0,
                                               size=len(val_positions))
            val_positions = np.clip(val_positions, 0, n_remaining - 1).astype(int)
            selected_val = remaining_indices[val_positions]
            val_indices.extend(selected_val)

            val_mask = np.ones(n_remaining, dtype=bool)
            val_mask[val_positions] = False
            train_indices.extend(remaining_indices[val_mask])
        else:
            train_indices.extend(remaining_indices)

    test_indices = np.array(test_indices)
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)

    test_files = files[test_indices]
    test_labels = labels[test_indices]

    val_files = files[val_indices]
    val_labels = labels[val_indices]

    train_files = files[train_indices]
    train_labels = labels[train_indices]

    return train_files, val_files, test_files, train_labels, val_labels, test_labels

    



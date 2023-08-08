import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
# from densephrases import DensePhrases
from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
from simcse import SimCSE
from torch import nn, optim, sum as tsum, reshape
from torch.utils.data import DataLoader, Dataset

from utils_phrasebert import load_model
from datetime import datetime


def get_data_emb(full_run_mode, task, split, model_path, device, shuffle=True, contextual=False):

    if task == "phrase_similarity":
        # Train + Evaluate on PS-hard
        dataset_path = "PiC/phrase_similarity"
        data_list = load_dataset(dataset_path)[split]   # download_mode="force_redownload"
    else:
        print("Task {} is currently not supported.".format(task))
        return
    
    phrase1_list = [item['phrase1'] for item in data_list]
    phrase2_list = [item['phrase2'] for item in data_list]
    labels = [item['label'] for item in data_list]

    context1_list = [item['sentence1'] for item in data_list] if contextual else []
    context2_list = [item['sentence2'] for item in data_list] if contextual else []

    if not full_run_mode:
        subset_size = 50
        phrase1_list = phrase1_list[:subset_size]
        phrase2_list = phrase2_list[:subset_size]
        labels = labels[:subset_size]
    
    model = load_model(model_path, device)

    if isinstance(model, DensePhrases) or isinstance(model, SimCSE):
        model.model.to(device)
    elif isinstance(model, SentenceTransformer):
        model.to(device)

    print(device)
    emb_batch_size = 32

    if not contextual:
        phrase1_emb_tensor_list = encode_in_batch(model, emb_batch_size, phrase1_list, device)
        phrase2_emb_tensor_list = encode_in_batch(model, emb_batch_size, phrase2_list, device)
    else:
        phrase1_emb_tensor_list = encode_with_context_in_batch(model, emb_batch_size, phrase1_list, context1_list, device)
        phrase2_emb_tensor_list = encode_with_context_in_batch(model, emb_batch_size, phrase2_list, context2_list, device)

    combined_phrase_list = []
    for phrase1_emb_tensor, phrase2_emb_tensor, label in zip(phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels):
        if phrase1_emb_tensor.shape[0] > 0 and phrase2_emb_tensor.shape[0] > 0:
            combined_phrase_list.append((phrase1_emb_tensor, phrase2_emb_tensor, label))

    phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels = zip(*combined_phrase_list)
    assert len(phrase1_emb_tensor_list) == len(phrase2_emb_tensor_list)

    if shuffle:
        import random
        random.seed(42)
        combined = list(zip(phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels))
        random.shuffle(combined)
        phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels = zip(*combined)

    label_tensor = torch.FloatTensor(labels)
    
    return torch.stack(phrase1_emb_tensor_list), torch.stack(phrase2_emb_tensor_list), label_tensor


def encode_in_batch(model, batch_size, text_list, device):
    all_emb_tensor_list = []

    if isinstance(model, DensePhrases) or isinstance(model, SimCSE):
        model.model.eval()
    elif isinstance(model, SentenceTransformer):
        model.eval()

    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_text_list = text_list[i:i+batch_size]
            if isinstance(model, DensePhrases):
                inputs = model.tokenizer(batch_text_list, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
                embeddings = model.model.embed_phrase(**inputs)[0]

                batch_emb_list = []
                for emb_idx, token_embeddings in enumerate(embeddings):
                    att_mask = list(inputs['attention_mask'][emb_idx])
                    last_token_index = att_mask.index(0) - 1 if 0 in att_mask else len(att_mask) - 1
                    batch_emb_list.append(token_embeddings[:last_token_index + 1].mean(dim=0))  # mean_pooling

            elif isinstance(model, SimCSE):
                inputs = model.tokenizer(batch_text_list, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
                outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)

                batch_emb_list = []
                for idx, token_embeddings in enumerate(outputs.last_hidden_state):
                    att_mask = list(inputs['attention_mask'][idx])
                    last_token_index = att_mask.index(0) - 1 if 0 in att_mask else len(att_mask) - 1
                    batch_emb_list.append(token_embeddings[:last_token_index + 1].mean(dim=0))

            elif isinstance(model, SentenceTransformer):
                batch_emb_list = model.encode(batch_text_list, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

            all_emb_tensor_list.extend(batch_emb_list)

    return all_emb_tensor_list


def encode_with_context_in_batch(model, batch_size, text_list, context_list, device):
    all_contextual_emb_tensor_list = []

    if isinstance(model, DensePhrases) or isinstance(model, SimCSE):
        model.model.eval()
    else:
        model.eval()

    with torch.no_grad():
        for i in range(0, len(context_list), batch_size):
            batch_text_list = text_list[i:i + batch_size]
            batch_context_list = context_list[i:i + batch_size]

            if isinstance(model, DensePhrases):
                inputs = model.tokenizer(batch_context_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
                batch_emb_list = model.model.embed_phrase(**inputs)[0]
            elif isinstance(model, SimCSE):
                inputs = model.tokenizer(batch_context_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
                outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)
                batch_emb_list = outputs.last_hidden_state
            else:
                batch_emb_list = model.encode(batch_context_list, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False, output_value="token_embeddings")

            contextual_phrase_embs = extract_contextual_phrase_embeddings(model, batch_text_list, batch_context_list, batch_emb_list)
            all_contextual_emb_tensor_list.extend(contextual_phrase_embs)

    return all_contextual_emb_tensor_list


def extract_contextual_phrase_embeddings(model, list_phrase, sentences, sentence_embeddings, max_length=256):

    def find_sub_list(sl, l):
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                return ind, ind + sll - 1

    all_phrase_embs = []
    if isinstance(model, DensePhrases):
        max_seq_length = model.config.max_position_embeddings
    elif isinstance(model, SimCSE):
        max_seq_length = model.model.config.max_position_embeddings
    else:
        max_seq_length = model.get_max_seq_length()

    for idx, (phrase, sent) in enumerate(zip(list_phrase, sentences)):
        encoded_phrase = model.tokenizer.encode_plus(text=phrase, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)
        encoded_phrase = np.array(encoded_phrase["input_ids"])[np.array(encoded_phrase["attention_mask"]) == 1]

        encoded_sent = model.tokenizer.encode_plus(text=sent, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)
        encoded_sent = np.array(encoded_sent["input_ids"])[np.array(encoded_sent["attention_mask"]) == 1]

        try:
            start_idx, end_idx = find_sub_list(list(encoded_phrase[1:-1]), list(encoded_sent))
            if end_idx >= max_seq_length:
                print("Context is too long: Idx {} - Phrase: {} - Sentence: {}".format(idx, phrase, sent))
                all_phrase_embs.append(torch.FloatTensor(0))
                continue
        except:
            print("Phrase not found: Idx {} - Phrase: {} - Sentence: {}".format(idx, phrase, sent))
            all_phrase_embs.append(torch.FloatTensor(0))
            continue

        phrase_indices = list(range(start_idx, end_idx + 1, 1))

        if isinstance(model, DensePhrases):
            phrase_embs = sentence_embeddings[idx][phrase_indices]
            phrase_embs = phrase_embs.mean(dim=0)
        elif isinstance(model, SimCSE):
            phrase_embs = sentence_embeddings[idx][phrase_indices]
            phrase_embs = phrase_embs.mean(dim=0)
        else:
            phrase_embs = sentence_embeddings[idx][phrase_indices]
            phrase_embs = phrase_embs.mean(dim=0)

        all_phrase_embs.append(phrase_embs)

    return all_phrase_embs


class ParaphraseDataset(Dataset):
    def __init__(self, phrase1_tensor, phrase2_tensor, label_tensor):
        self.concat_input = torch.cat((phrase1_tensor, phrase2_tensor), 1)
        self.label = label_tensor

    def __getitem__(self, index):
        return (self.concat_input[index], self.label[index])

    def __len__(self):
        return self.concat_input.size()[0]


class ProbingModel(LightningModule):
    def __init__(self, input_dim=1536, train_dataset=None, valid_dataset=None, test_dataset=None):
        super(ProbingModel, self).__init__()
        # Network layers
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.output = nn.Sigmoid()

        # Hyper-parameters, that we will auto-tune using lightning!
        self.lr = 0.0001
        self.batch_size = 200

        # datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        # ThangPM: For saving predictions
        self.test_y = []
        self.test_y_hat = []

    def forward(self, x):
        x1 = self.linear(x)
        x1a = F.relu(x1)
        x2 = self.linear2(x1a)
        output = self.output(x2)
        return reshape(output, (-1,))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def compute_accuracy(self, y_hat, y):
        with torch.no_grad():
            y_pred = (y_hat >= 0.5)
            y_pred_f = y_pred.float()
            num_correct = tsum(y_pred_f == y)
            denom = float(y.size()[0])
            accuracy = torch.div( num_correct, denom)
        return accuracy

    def training_step(self, batch, batch_nb):
        mode = 'train'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'loss': loss, f'{mode}_accuracy':accuracy, 'log': {f'{mode}_loss': loss}}

    def training_epoch_end(self, outputs):
        mode = 'train'
        loss_mean = sum([o[f'loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

    def validation_step(self, batch, batch_nb):
        mode = 'val'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'{mode}_loss': loss, f'{mode}_accuracy':accuracy, 'log': {f'{mode}_loss': loss}}

    def validation_epoch_end(self, outputs):
        mode = 'val'
        loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

    def test_step(self, batch, batch_nb):
        mode = 'test'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)

        # ThangPM: Save predictions
        self.test_y.extend(y)
        self.test_y_hat.extend(y_hat)

        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'{mode}_loss': loss, f'{mode}_accuracy':accuracy, 'log': {f'{mode}_loss': loss}}

    def test_epoch_end(self, outputs):
        mode = 'test'
        loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

        # ThangPM: Check qualitative examples for debugging purposes
        with open("../results/predictions_{}.txt".format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')), "w") as output_file:
            [output_file.write("Idx {}: GT: {} -- Pred: {} -- Conf: {:.4f}\n".format(idx, y, (y_hat >= 0.5).float(), y_hat))
             for idx, (y, y_hat) in enumerate(zip(self.test_y, self.test_y_hat))]


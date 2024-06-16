import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import re
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import random
from seq2seq import TextDataset, load_corpus

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size, max_len=5000, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.create_positional_encoding(embed_size, max_len)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, ff_hidden_size, dropout)
        self.fc = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

    def create_positional_encoding(self, embed_size, max_len):
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pos_t = torch.tensor(pos, dtype=torch.float32)
                pe[pos, i] = torch.sin(pos_t / (10000 ** (i / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = torch.cos(pos_t / (10000 ** ((i + 1) / embed_size)))
        return pe.unsqueeze(0)

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.embed_size) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt = self.embedding(tgt) * math.sqrt(self.embed_size) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, embed_size)
        tgt = tgt.permute(1, 0, 2)  # (seq_length, batch_size, embed_size)
        output = self.transformer(src, tgt)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, embed_size)
        output = self.fc(output)
        return output

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        self.text_as_int = [self.char2idx[c] for c in text]
    
    def __len__(self):
        return len(self.text_as_int) - self.seq_length
    
    def __getitem__(self, idx):
        return (torch.tensor(self.text_as_int[idx:idx+self.seq_length]),
                torch.tensor(self.text_as_int[idx+1:idx+self.seq_length+1]))

def load_corpus():
    print('begin load corpus')
    inf = open("./datasets_cn/inf.txt", "r", encoding="gb18030").read()  # gb18030 utf-8
    inf = inf.split(',')
    corpus = []
    for name in tqdm(inf):
        with open("./datasets_cn/" + name + ".txt", "r", encoding="gb18030") as f:
            txt = f.read()
            ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
            txt = txt.replace(ad, '')
            txt = txt.replace(' ', '')
            txt = txt.replace('\n', '')
            txt = txt.replace('□', '')
            corpus += txt
    return corpus
def generate_text(model, start_string, gen_length, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    generate_text = start_string
    # i = 0
    while True:
        if len(generate_text) > gen_length:
            break
        generate_text_index = [dataset.char2idx[s] for s in generate_text]
        # input_index = generate_text_index[i:i+dataset.seq_length]
        # output = model(torch.tensor(input_index).unsqueeze(0).to(device), torch.tensor(input_index).unsqueeze(0).to(device))
        output = model(torch.tensor(generate_text_index[:-1]).unsqueeze(0).to(device), torch.tensor(generate_text_index[1:]).unsqueeze(0).to(device))
        top1 = output.argmax(2)[:, -1].item()
        next_char = dataset.idx2char[top1]
        generate_text += next_char
        # i += 1

    
    return generate_text

if __name__ == '__main__':
  
    # text = load_corpus()
    # seq_length = 30
    # dataset = TextDataset(text, seq_length)
    # torch.save(dataset, 'dataset.pth')
    # # dataset = torch.load('dataset10.pth')
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vocab_size = len(dataset.vocab)
    # print("vocab_size: ", vocab_size)
    # embed_size = 128
    # hidden_size = 256
    # num_layers = 2
    # num_heads = 8

    # model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, hidden_size).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # model_load = torch.load('transformer_8.pth')
    # # model.load_state_dict(model_load['model_state_dict'])
    # # optimizer.load_state_dict(model_load['optimizer_state_dict'])

    # start = 0
    # epochs = 50
    # loss_list = []
    # model.train()
    # for epoch in range(start, epochs):
    #     loss_sum = 0
    #     for source, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
    #         source, target = source.to(device), target.to(device)
    #         output = model(source, target)
    #         output = output.reshape(-1, output.shape[2])
    #         target = target.reshape(-1)
    #         loss = criterion(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    #         loss_sum += loss.item()
    #     loss_avg = loss_sum / len(dataloader)
    #     pickle.dump(loss_list, open('loss_list.pkl', 'wb'))
    #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_avg:.4f}")
    #     torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'transformer_{}.pth'.format(epoch+1))

    text = load_corpus()
    seq_length = 30
    dataset = TextDataset(text, seq_length)
    # dataset = torch.load('dataset10.pth')
    start_string = "他跨下的枣红马奔驰了数十里地，早已筋疲力尽，在主人没命价的鞭打催踢之下，逼得气也喘不过来了，这时嘴边已全是白沫，猛地里前腿一软，跪倒在地。那汉子用力一提缰绳"
    gen_length = 500
    
    vocab_size = len(dataset.vocab)
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    num_heads = 8
    seq_length = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, hidden_size).to(device)
    model.load_state_dict(torch.load('./model/transformer_5.pth')['model_state_dict'])

    generated_text = generate_text(model, start_string, gen_length, dataset)
    print(generated_text)




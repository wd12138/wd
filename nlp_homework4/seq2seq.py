import torch
import torch.nn as nn
import random
import os
import re
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x.unsqueeze(1))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(target.device)
        hidden, cell = self.encoder(source)

        input = target[:, 0] 
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = target[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        
        return outputs


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
    corpus = ''
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
    while True:
        if len(generate_text) > gen_length:
            break
        generate_text_index = [dataset.char2idx[s] for s in generate_text]
        output = model(torch.tensor(generate_text_index[:-1]).unsqueeze(0).to(device), torch.tensor(generate_text_index[1:]).unsqueeze(0).to(device), 0)
        top1 = output.argmax(2)[:, -1].item()
        next_char = dataset.idx2char[top1]
        generate_text += next_char

    
    return generate_text

if __name__ == '__main__':
    # text = load_corpus()
    # seq_length = 30
    # dataset = TextDataset(text, seq_length)
    # torch.save(dataset, 'dataset.pth')
    # # dataset = torch.load('dataset.pth')
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vocab_size = len(dataset.vocab)
    # print("vocab_size: ", vocab_size)
    # embed_size = 128
    # hidden_size = 256
    # num_layers = 2

    # encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # model = Seq2Seq(encoder, decoder).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # start = 0
    # epochs = 1
    # loss_list = []
    # model.train()
    # for epoch in range(start, epochs):
    #     loss_sum = 0
    #     for source, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
    #         source, target = source.to(device), target.to(device)
    #         output = model(source, target)
    #         output = output[:, 1:].reshape(-1, output.shape[2])
    #         target = target[:, 1:].reshape(-1)
    #         loss = criterion(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    #         loss_sum += loss.item()
    #     loss_avg = loss_sum / len(dataloader)
    #     pickle.dump(loss_list, open('loss_list.pkl', 'wb'))
    #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_avg:.4f}")
    #     torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'seq2seq_{}.pth'.format(epoch+1))

    # text = load_corpus()
    # seq_length = 30
    # dataset = TextDataset(text, seq_length)
    # torch.save(dataset, 'dataset.pth')
    # # dataset = torch.load('dataset.pth')
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vocab_size = len(dataset.vocab)
    # print("vocab_size: ", vocab_size)
    # embed_size = 128
    # hidden_size = 256
    # num_layers = 2

    # encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # model = Seq2Seq(encoder, decoder).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # start = 0
    # epochs = 50
    # loss_list = []
    # model.train()
    # for epoch in range(start, epochs):
    #     loss_sum = 0
    #     for source, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
    #         source, target = source.to(device), target.to(device)
    #         output = model(source, target)
    #         output = output[:, 1:].reshape(-1, output.shape[2])
    #         target = target[:, 1:].reshape(-1)
    #         loss = criterion(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    #         loss_sum += loss.item()
    #     loss_avg = loss_sum / len(dataloader)
    #     pickle.dump(loss_list, open('loss_list.pkl', 'wb'))
    #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_avg:.4f}")
    #     torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'seq2seq_{}.pth'.format(epoch+1))

    text = load_corpus()
    seq_length = 30
    dataset = TextDataset(text, seq_length)
    start_string = "他跨下的枣红马奔驰了数十里地，早已筋疲力尽，在主人没命价的鞭打催踢之下，逼得气也喘不过来了，这时嘴边已全是白沫，猛地里前腿一软，跪倒在地。那汉子用力一提缰绳"
    gen_length = 500
    
    vocab_size = len(dataset.vocab)
    embed_size = 128
    hidden_size = 256
    num_layers = 2
   # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' )
    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load('./model/seq2seq_1.pth')['model_state_dict'])

    generated_text = generate_text(model, start_string, gen_length, dataset)
    print(generated_text)





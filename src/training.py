import pandas as pd
import torch
import torch.nn.functional as F
import random
import time

from model import Linear, BatchNorm1d, Tanh

EPOCHS = 100000
BATCH_SIZE = 32

N_EMBED = 10 # dimension of the character embedding vectors
N_HIDDEN = 100 # number of neurons in the hidden layer of MLP
BLOCK_SIZE = 3 # Context length
SAMPLES = 1000

LAYERS = []

def format_dataset():
    data = pd.read_csv('../dataset/indian-male-names.csv')
    indian_names = data['name']
    names = indian_names.to_list()

    return remove_invalid_names(names)

def build_dataset(names, s_toi):
    X, Y = [], []

    for name in names:
        context = [0] * BLOCK_SIZE
        
        for ch in name + '+':
            ix = s_toi[ch]
            X.append(context)
            Y.append(ix)
            # crop the first latter and append the one ahead
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

def build_neural_net(embeddings, vocab_size):
    layers = [
        Linear(N_EMBED * BLOCK_SIZE, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(            N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(            N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(            N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(            N_HIDDEN, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(            N_HIDDEN, vocab_size, bias=False), BatchNorm1d(vocab_size),
    ]

    with torch.no_grad():
        # last layer: make less confident
        layers[-1].gamma *= 0.1
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 1.0
    
    parameters = [embeddings] + [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True

    return layers, parameters


def train_model(X_tr, Y_tr, layers, parameters, embeddings):
    for i in range(EPOCHS):
        # mini-batch construct
        ix = torch.randint(0, X_tr.shape[0], (BATCH_SIZE,))
        Xb, Yb = X_tr[ix], Y_tr[ix] # batch X,Y
  
        # forward pass
        emb = embeddings[Xb] # embed the characters into vectors
        x = emb.view(emb.shape[0], -1) # concatenate the vectors
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb) # loss function
  
        # backward pass
        for layer in layers:
            layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
        
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # update
        lr = 0.01 # if i < 150000 else 0.01 # step learning rate decay
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i % 1000 == 0: # print every once in a while
            print(f'{i:7d}/{EPOCHS:7d}: {loss.item():.4f}')

def inference(layers, i_tos, embeddings):
    # sample from the model
    generated_names = []

    for _ in range(SAMPLES):
    
        out = []
        context = [0] * BLOCK_SIZE # initialize with all ...
        while True:
            # forward pass the neural net
            emb = embeddings[torch.tensor([context])] # (1,block_size,n_embd)
            x = emb.view(emb.shape[0], -1) # concatenate the vectors
            
            for layer in layers:
                x = layer(x)
            
            logits = x
            probs = F.softmax(logits, dim=1)
            
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            
            # shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            
            # if we sample the special '.' token, break
            if ix == 0:
                break
        
        generated_names.append(''.join(i_tos[i] for i in out)[:-1])
        # decode and print the generated word
        # print(''.join(i_tos[i] for i in out)[:-1])

    return generated_names


def execute():
    # Get formatted Indian Male names
    indian_names = format_dataset()
    s_toi, i_tos, vocab_size = build_character_mappings(indian_names)
    
    random.shuffle(indian_names)

    # 80%, 10%, 10% Train, Dev and Test splits
    n1 = int(0.8 * len(indian_names))
    n2 = int(0.9 * len(indian_names))

    X_tr, Y_tr = build_dataset(indian_names[:n1], s_toi)
    X_dev, Y_dev = build_dataset(indian_names[n1:n2], s_toi)
    X_test, Y_test = build_dataset(indian_names[n2:], s_toi)

    embeddings = torch.randn((vocab_size, N_EMBED))

    layers, parameters = build_neural_net(embeddings, vocab_size)
    train_model(X_tr, Y_tr, layers, parameters, embeddings)

    # put layers into eval mode
    for layer in layers:
        layer.training = False

    evaluate_loss('dev', X_dev, Y_dev, layers, embeddings)
    evaluate_loss('test', X_test, Y_test, layers, embeddings)
    
    generated_names = inference(layers, i_tos, embeddings)
    command_line_print(generated_names)
    save_as_csv(generated_names)


def command_line_print(generated_names):
    print('Model generating more Indian names...')
    for generated_name in generated_names:
        print(generated_name)
        time.sleep(1)


def save_as_csv(generated_names):
    df = pd.DataFrame(generated_names, columns=['Name'])
    df.to_csv('../inference/generation.csv')


def evaluate_loss(split, x, y, layers, embeddings):
    emb = embeddings[x] # (N, block_size, n_embd)
    x = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
    
    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, y)
    print(split, loss.item())


def build_character_mappings(names):
    chars = sorted(list(set(''.join(names))))
    s_toi = {s:i+1 for i,s in enumerate(chars)}
    s_toi['+'] = 0

    i_tos = {i:s for s,i in s_toi.items()}
    return s_toi, i_tos, len(chars) + 1


def remove_invalid_names(names):
    mod_names = []
    for name in names:
        if type(name) is str:
            mod_names.append(name)
    
    return mod_names


# Execute name generation
execute()
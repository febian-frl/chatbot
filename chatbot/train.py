import itertools
import random
import torch.nn as nn
from torch import optim
from load import loadPrepareData
from model import EncoderRNN, LuongAttnDecoderRNN
from config import *
import matplotlib.pyplot as plt


def draw(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss figure')
    plt.savefig('loss.jpg')



# 把句子的词变成ID
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# l是多个长度不同句子(list)，使用zip_longest padding成定长，长度为最长句子的长度。
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# l是二维的padding后的list
# 返回m和l的大小一样，如果某个位置是padding，那么值为0，否则为1
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# 把输入句子变成ID，然后再padding，同时返回lengths这个list，标识实际长度。
# 返回的padVar是一个LongTensor，shape是(batch, max_length)，
# lengths是一个list，长度为(batch,)，表示每个句子的实际长度。
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# 对输出句子进行padding，然后用binaryMatrix得到每个位置是padding(0)还是非padding，
# 同时返回最大最长句子的长度(也就是padding后的长度)
# 返回值padVar是LongTensor，shape是(batch, max_target_length)
# mask是BoolTensor，shape也是(batch, max_target_length)
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


"""
处理一个batch的pair句对.
input_variable的每一列表示一个样本,每一行表示pair_batch个样本在这个时刻的值。
lengths表示真实的长度。
类似的target_variable也是每一列表示一个样本，mask的shape和target_variable一样，如果某个位置是0，则表示padding。
"""


def batch2TrainData(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer):
    # 梯度清空
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 数据放入gpu或cpu
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths 参数始终放在cpu上处理
    lengths = lengths.to("cpu")

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # 将参数传入编码器前向传播
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # 创建解码所需的输入，从每段语句的embeddingSOS tokens开始
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device) # 放入设备中

    # 初始化所有解码隐层单元
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 是否使用teacher_forcing_ratio强制本次迭代
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 解码器一次一步地传入序列
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 当前输出是下一步的输入
            decoder_input = target_variable[t].view(1, -1)
            # 计算损失
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 下一步的输入是解码器自身输出
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 计算损失
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    loss.backward()

    # 梯度裁剪，防止梯度爆炸或消失
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 参数更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, corpus_name):
    # 加载batch_size数量数据
    reverse = None
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                        for _ in range(n_iteration)]
    # 初始化
    checkpoint = None
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # 开始迭代训练
    iter_list = []
    loss_list = []
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # 提取字段
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 批处理迭代训练
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                     encoder_optimizer, decoder_optimizer)
        print_loss += loss

        # 输出进度
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0
            iter_list.append(iteration)
            loss_list.append(print_loss_avg)
            draw(iter_list, loss_list)

        # 保存模型
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def run(run_train, run_test):
    # 句对重新规整化 将词语转换成稠密词向量
    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        embedding.load_state_dict(embedding_sd)
    # 定义encoder、decoder网络，编解码模型实例化
    encoder = EncoderRNN(voc.num_words, hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # 有gpu尽量用gpu加载模型，没有就用cpu
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if run_train:
        # 开启训练模式
        encoder.train()
        decoder.train()

        # 定义优化器
        encoder_optimizer = optim.NAdam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.NAdam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

        if loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, corpus_name)

    elif run_test:
        encoder.eval()
        decoder.eval()
        from model import GreedySearchDecoder
        from evaluate1 import evaluateInput
        searcher = GreedySearchDecoder(encoder, decoder)
        evaluateInput(encoder, decoder, searcher, voc)


if __name__ == '__main__':
    run(run_train=True, run_test=False)

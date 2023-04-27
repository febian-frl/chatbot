from train import indexesFromSentence
from config import *


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # 词转换成索引
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建lengths张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转换batch的维度，转置矩阵
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # 解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    # 索引转换成词
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = input('【user】> ')
    from load import normalizeString
    input_sentence = normalizeString(input_sentence)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('【Bot:】', ' '.join(output_words))

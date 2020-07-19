from word_sequence import WordSequence
from dataset import get_dataloader
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    ws = WordSequence()
    train_data = get_dataloader(train=True)
    test_data = get_dataloader(train=False)
    for content,labels in tqdm(train_data):
        # content是包含batch_size个评论
        for sentence in content:
            # 计算词频
            ws.fit(sentence)
    # 开始构建词典
    ws.build_vocab(min_count=5,max_count=10000)
    print(len(ws))
    # dump jiang数据通过特殊的形式转换为只有python语言认识的字符串，并写入文件
    pickle.dump(ws,open("./model/ws.pkl","wb"))


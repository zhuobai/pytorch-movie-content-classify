from torch.utils.data import DataLoader,Dataset
import torch
import os
import re   #对字符串进行操作
from word_sequence import WordSequence
import lib
def tokenlize(content):
    '''
    函数功能：将一个句子拆分成一个个单词列表
    :param content:
    :return:
    1、先说sub是替换的意思。
    2、.是匹配任意字符（除换行符外）*是匹配前面的任意字符一个或多个
    3、？是非贪婪。
    4、组合起来的意思是将"<"和中间的任意字符">" 换为空字符串""
    由于有？是非贪婪。 所以是匹配"<"后面最近的一个">"
    '''
    content = re.sub("<.*?>","*",content)
    filters = [':','\t','$','%','#','&','\n','\x96','\x97','\.']
    # join函数是用来作分割符的
    content = re.sub("|".join(filters)," ",content)
    # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self,train=True):
        super(ImdbDataset, self).__init__()
        data_path = r"E:\深度学习资料\aclImdb"
        data_path += r"\train" if train else r"\test"
        self.total_path = []
        for temp_path in [r"\pos",r"\neg"]:
            cur_path = data_path + temp_path
            # 添加积极和消极评论的所有文件
            self.total_path += [os.path.join(cur_path,i) for i in os.listdir(cur_path) if i.endswith(".txt")]

    def __getitem__(self, idx):
        # 获取某个评论的文件路径
        file = self.total_path[idx]
        # 读取评论内容
        content = open(file=file,encoding="utf-8").read()
        # 将评论分成一个个单词列表
        content = tokenlize(content)
        # 获取评论的分数（小于5为消极，大于等于5为积极）
        label = int(file.split("_")[1].split(".")[0])
        # 设置消极评论为0，积极评论为1
        label = 0 if label<5 else 1
        return content,label

    def __len__(self):
        # 返回所有文件的个数
        return len(self.total_path)

def collate_fn(batch):
    """
    对batch数据进行处理([tokens,label],[tokens,label]...)
    :param batch:
    :return:
    """
    # *batch 可理解为解压，返回二维矩阵式
    content,labels = list(zip(*batch))
    # content中是有batch_size个评论（句子）
    content = [lib.ws.transform(sentence,200) for sentence in content]
    # content式字符串数组，必须先将数组中字符转化成对应数字，才能转成张量
    content = torch.LongTensor(content)
    labels = torch.LongTensor(labels)

    return content,labels

def get_dataloader(train=True):
    imdbdataset = ImdbDataset(train)

    return DataLoader(imdbdataset,batch_size=128,shuffle=True,collate_fn=collate_fn)


if __name__ == '__main__':
    # dataset = ImdbDataset(True)
    # print(dataset[0])
    # print(len(get_dataloader(True)))
    # print(type(get_dataloader()))
    # exit()
    print(get_dataloader())
    exit()
    for idx, (content,label) in enumerate(get_dataloader(True)):
        # for i,con in enumerate(content,0):
        #     print(i,con)
        print(idx)
        print(content)
        print(label)
        break



import gensim.downloader as api
from gensim.models import Word2Vec

'''
下载所需要的词典，这里使用的是text8，也可以使用glove-twitter-25
gensim-data里有已经训练好的词向量字典
这里是保存为二进制文件
'''
# diction = "text8"
diction = "glove-twitter-25"
info = api.info(diction)
assert(len(info) > 0)
print(info)
dataset = api.load("text8",return_path=True)
model = Word2Vec(dataset)

model.save("data/text8-word2vec.bin")

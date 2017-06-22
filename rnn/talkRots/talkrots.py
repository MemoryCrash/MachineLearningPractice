#!/usr/bin/env python
# -*-coding:UTF-8 -*-
# coding:utf-8

import pynlpir

pynlpir.open()

s = '怎么才能把电脑里的垃圾文件删除'

key_words = pynlpir.get_key_words(s, weighted=True)
for key_word in key_words:
    print("{}\t{}".format(key_word[0], key_word[1]))

pynlpir.close()
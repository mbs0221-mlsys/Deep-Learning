import pkuseg

seg = pkuseg.pkuseg()
text = seg.cut('对input.txt的文件分词输出到output.txt中，使用默认模型和词典，开20个进程')
print(text)
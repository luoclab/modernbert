import re

def data_pro(content):
    """
    该函数主要用于：

    去除特殊字符（如 "…" 和 ".."）
    保留中文、英文、数字和特定符号
    转换英文字符为小写
    """
    content = content.replace('…', '').replace('..','')
    content = content.replace("\n", '')#换行符换成空格
    text = ''
    for n in range(0, len(content)-1):
#         if '\u4e00' <= content[n] <= '\u9fff' or content[n] in '。？！，；：‘“”’/（）《》.0123456789.%':
        if '\u4e00' <= content[n] <= '\u9fff' or content[n] in '@$*&!,。？！，；：:-_‘“”’/（）《》.0123456789.% '\
                    or '\u0041' <= content[n] <= '\u005A' or '\u0061' <= content[n] <= '\u007A':
            text += content[n]
    text = text.lower()
    return text


if __name__ == '__main__':
    content = "Hello! 你好…这是一个测试… 价格是$100.5。"
    print(data_pro(content))

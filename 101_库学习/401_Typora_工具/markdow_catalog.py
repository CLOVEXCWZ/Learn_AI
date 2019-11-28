import re
import os


class MarkdowCatalog(object):

    def __init__(self, file_path):
         self.file_path = file_path;

    def opt_markdown(self, opt_title_index=True):
        # 优化markdown,添加目录以及添加索引
        if opt_title_index:
            lines = self.opt_title_index(save=False)
        else:
            with open(self.file_path, 'r') as f:
                lines = f.read().split('\n')

        line_optimzse = []
        title_lines = []
        code_flag = 0
        for line in lines:
            # 处理代码
            if str(line).startswith("```"):
                code_flag += 1
            if code_flag%2==0 and self.__is_title(line):  # 是标题需要优化
                name, pre = self.__title_split(line)
                leve = len(re.search('#+', pre).group(0))  # 标题级别
                name = str(name).strip()
                title_optimze = f'{pre} <a name="{name}">{name}</a>'
                line_optimzse.append(title_optimze)
                table = "".join(['\t'] * (leve - 1))
                title_link = f"{table}- [{name}](#{name})"
                title_lines.append(title_link)
            else:
                line_optimzse.append(line)
        if len(title_lines) > 0:
            title_lines[0] = str(title_lines[0]).strip()
        line_optimzse = ["**目录**\n"] + title_lines + ["\n\n\n"] + line_optimzse
        self.__save_to_file(line_optimzse)

    def opt_title_index(self, save=True, re_index=True):
        """
        优化文档标题的index
        :param save: 是否标题
        :param re_index: 是否重先index
        :return: 返回优化后的文档内容
        """
        with open(self.file_path, 'r') as f:
            lines = f.read().split('\n')
        line_optimzse = []
        title_index = "0"

        code_flag = 0
        for line in lines:
            # 处理代码
            if str(line).startswith("```"):
                code_flag += 1
            if code_flag%2==0 and self.__is_title(line):  # 是标题需要优化
                name, pre = self.__title_split(line)
                leve = len(re.search('#+', pre).group(0))  # 标题级别
                name = str(name).strip()
                if re_index:
                    name = self.__delet_title_index(name)
                title_index = self.__add_title_index(title_index, leve)
                title_optimze = f'{pre} {title_index} {name}'
                line_optimzse.append(title_optimze)
            else:
                line_optimzse.append(line)
        if save:
            self.__save_to_file(line_optimzse, add_name='_opt_index')
        return line_optimzse

    def __add_title_index(self, curent_index, leve):
        # index 加1的情况 比如 add_title_index("1.2.4", 4) -> 1.2.4.1
        # add_title_index("1.2.4", 1) -> 2
        index_list = str(curent_index).split('.')
        index_list = [int(item) for item in index_list]

        if len(index_list) == leve:
            pass
        elif len(index_list) > leve:
            index_list = index_list[:leve]
        else:
            index_list += [0]
        index_list[-1] = index_list[-1] + 1
        index_list = [str(item) for item in index_list]
        return ".".join(index_list)

    def __save_to_file(self, text_lines, add_name="_opt"):
        path, extension = os.path.splitext(self.file_path)
        save_path = f"{path}{add_name}{extension}"
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(text_lines))
        print(f"文档保存至:{save_path}")

    def __is_title(self, line):
        # 判断是否是标题
        return bool(re.match('^[-*\s]*#+', line))

    def __title_split(self, title, delete_link=True):
        """
        提取标题的名称和前缀
        通过正则表达式匹配标题
        :param title:
        :param delete_link 删除出原来的连接
        :return: name 标题  pre 标题的前缀
        """
        span_range = re.match('^[-*\s]*#+', title).span() # 直接匹配标题(用于分离标题名称和前缀)
        pre = title[:span_range[1]]
        name = title[span_range[1]:]

        if delete_link: # 删除原来的索引
            # 判断是否已经由索引
            if re.search(".*<.*>.*</.*>.*", name):
                #如果已经由索引，去除索引
                name_ = re.search(">.*</", name).group(0)
                name = name_[1:-2]
        return name, pre

    def __delet_title_index(self, title):
        """
        删除标题的 index
        采用正则表达式匹配去判断是否存在标题index，如果存在采用正则匹配，并删除标题
        :param title: 标题 (这里将不做标题判断)
        :return: 去除 index 之后的标题
        """
        if re.match("^\s*([0-9]*[.]*)+", title): # 判断是否存在Index
            span_range = re.match("^\s*([0-9]*[.]*)+", title).span() # 查找出index 的范围
            return title[span_range[1]:] # 删除标题的index
        else:
            return title


if __name__ == '__main__':
    file_path="""
    /Users/zhouwencheng/Desktop/Life/601AI/101_库学习/202_Pytorch/201_练习/101_model/901_torch_Module探索学习报告.md
    """
    file_path = file_path.strip()
    opt = MarkdowCatalog(file_path=file_path)
    opt.opt_markdown(opt_title_index=True)
    # opt.opt_title_index()



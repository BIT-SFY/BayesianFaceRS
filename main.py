import os
import bayes
import train
import threading
import tkinter.ttk
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


def make_sure_database():
    '''
    确定待预测人脸的数据库
    :return:
    '''
    # 创建提示标签
    word3 = tk.Label(root, text="请选择数据库：", font=('宋体', '20'))
    word3.place(x=0, y=0)
    # 这是tkinter自带的一种字符串类型，以保证值的变更随时可以显示在界面上
    var = tk.StringVar()
    # 创建下拉列表
    combobox = tk.ttk.Combobox(root, textvariable=var, font=('宋体', '18'))
    # 获取人脸数据库列表
    file_list = os.listdir('./TrainingInfo')
    # 将数据库内容添入下拉列表中
    combobox['value'] = file_list
    # 设置下拉列表的默认选项
    combobox.current(0)
    # 下拉列表的宽度
    combobox.config(width=7)
    # 下拉列表的位置
    combobox.place(x=0, y=40)
    # 确认按钮
    button = tk.Button(root, text='确认', command=lambda: select_files(var.get()), font=('宋体', '15'))
    # 按钮位置
    button.place(x=120, y=40)
    return None


def select_files(dataBaseName):
    '''
    选择待识别人脸
    :param dataBaseName:对应人脸数据库名称
    :return:
    '''
    # askopenfilenames函数选择文件，获取文件路径
    selected_files_path = filedialog.askopenfilenames()
    # 打开图片 并使尺寸定为380*380
    img1 = Image.open(selected_files_path[0]).resize((380, 380))
    # 全局变量，以防图片被清理
    global photo1, photo2
    # 使用ImageTk的PhotoImage方法读取图片
    photo1 = ImageTk.PhotoImage(img1)
    # 确定图像的位置
    tk.Label(master=root, image=photo1).place(x=180, y=50)
    # 展示文字
    word1 = tk.Label(root, text="你选择的人像：", font=('宋体', '20'))
    word1.place(x=280, y=5)
    word2 = tk.Label(root, text="预测的人像是：", font=('宋体', '20'))
    word2.place(x=680, y=5)
    # 调用权重贝叶斯分类得到预测图像的类别
    preNum, timing = bayes.Prediction(selected_files_path[0], dataBaseName)
    # 得到运行时间,并保留4位小数
    timing = round(timing, 4)
    rootdir = './FaceLib/' + dataBaseName
    # 获取识别结果所属类别的第一张图像作为参考,包含戳对FERET数据库的特殊操作
    if preNum <= 9:
        preNum = str(preNum)
        if dataBaseName == 'FERET':
            preNum = '00' + preNum
            pass
        else:
            preNum = '0' + preNum
        pass
    elif preNum <= 99 and dataBaseName == 'FERET':
        preNum = str(preNum)
        preNum = '0' + preNum
        pass
    # 构建待分类图片的路径
    path1 = os.listdir(rootdir + '/s' + str(preNum))[0]
    # 根据图片路径打开图片
    img2 = Image.open(rootdir + '/s' + str(preNum) + '/' + path1).resize((380, 380))
    # 使用ImageTk的PhotoImage打开图像
    photo2 = ImageTk.PhotoImage(img2)
    tk.Label(master=root, image=photo2).place(x=580, y=50)
    # 显示用时
    word3 = tk.Label(root, text="耗时:{}s".format(timing), font=('宋体', '20'))
    word3.place(x=10, y=435)
    return None


def choose():
    '''
    弹出参数选择窗口，进行参数选择
    :return:None
    '''
    master = Toplevel()  # 创建顶层子窗口
    master.title('参数设置')  # 子窗口标题
    master.geometry('270x230+600+400')  # 子窗口哦的位置和大小

    # 设置标签实现文本显示
    tk.Label(master, text="PCA特征维数").grid(row=0)
    tk.Label(master, text="最大演化代数").grid(row=1)
    tk.Label(master, text="种群大小").grid(row=2)
    tk.Label(master, text="变异因子F(0,1)").grid(row=3)
    tk.Label(master, text="交叉概率CR(0,1)").grid(row=4)
    tk.Label(master, text="测试集比例(0,1)").grid(row=5)

    # 设置输入框获取输入
    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    e3 = tk.Entry(master)
    e4 = tk.Entry(master)
    e5 = tk.Entry(master)
    e6 = tk.Entry(master)

    # 设置输入框的默认值
    e1.insert(0, '20')
    e2.insert(0, '50')
    e3.insert(0, '20')
    e4.insert(0, '0.8')
    e5.insert(0, '0.5')
    e6.insert(0, '0.3')

    # 对输入框进行排版
    e1.grid(row=0, column=1, padx=10, pady=5)
    e2.grid(row=1, column=1, padx=10, pady=5)
    e3.grid(row=2, column=1, padx=10, pady=5)
    e4.grid(row=3, column=1, padx=10, pady=5)
    e5.grid(row=4, column=1, padx=10, pady=5)
    e6.grid(row=5, column=1, padx=10, pady=5)

    # 点击确认按钮后进行文件夹选择
    B1 = tk.Button(master, text="确认", width=10,
                   command=lambda: select_folder(master, e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get()))
    B1.grid(row=6, column=0, sticky="w", padx=10, pady=5)  # 对按钮进行排版
    tk.Button(master, text="退出", width=10, command=master.destroy).grid(row=6, column=1, sticky="e", padx=10,
                                                                        pady=5)  # 退出按钮的设置
    master.mainloop()
    return None


def select_folder(master, n_components, MAX_GENERATION, N, F, CR, test_size):
    '''
    选择人脸图像库进行分类器的训练
    :param master:传入的子窗口
    :param n_components:PCA特征向量维数
    :param MAX_GENERATION:最大演化代数
    :param N:种群大小
    :param F:变异因子
    :param CR:交叉概率
    :param test_size:测试集所占比重
    :return:None
    '''
    master.destroy()  # 参数传递成功后销毁参数设置窗口
    # 人脸图像库的选择
    selected_folder = filedialog.askdirectory()  # 使用askdirectory函数选择文件夹
    tk.messagebox.askokcancel('确认是否训练', '您是否要训练{}下的人脸数据库'.format(selected_folder))  # 确认提示窗，返回值true/false
    # 将人脸信息库地址和相关参数传入
    picNum, lableNum, timing, Accuracy = train.Fit(selected_folder, int(n_components), int(MAX_GENERATION), int(N),
                                                   float(F), float(CR), float(test_size))  # 进行分类器训练
    # 将得到的耗时进行截取
    timing = timing[:6]
    # 准确率保留两位小数
    Accuracy = round(Accuracy, 2)
    # 完成提示
    tk.messagebox.showinfo('完成提示', '分类器训练完成\n共{}张照片，{}类\n用时：{}秒\n准确度：{}%'.format(picNum, lableNum, timing, Accuracy))
    return None


root = tk.Tk()
# 画布设置
root.title("基于贝叶斯的人脸识别系统")  # 窗口标题
root.geometry("1000x480+380+250")  # 设置窗口大小 注意：是x 不是*
root.resizable(width=False, height=False)  # 设置窗口是否可以变化长/宽，False不可变，True可变，默认为True
root.tk.eval('package require Tix')  # 引入升级包，这样才能使用升级的组合控件
# 创建画布并布局
mycanvas1 = tk.Canvas(root, width=2, height=480, bg="white")
mycanvas1.place(x=570, y=0)
# 绘制线段
myline0 = mycanvas1.create_line(2, 0, 2, 480)
# 创建画布并布局
mycanvas2 = tk.Canvas(root, width=175, height=200, bg="white")
mycanvas2.place(x=0, y=150)
# 绘制线段
myline1 = mycanvas2.create_line(0, 2, 175, 2)
myline4 = mycanvas2.create_line(0, 100, 175, 100)
myline3 = mycanvas2.create_line(0, 200, 175, 200)
myline2 = mycanvas2.create_line(175, 0, 175, 200)
# 按钮控件
# 图像识别模块
Button1 = tk.Button(root, text="图像识别", font=('华文楷体', '20'), command=make_sure_database)
Button1.config(width=10, height=2)
Button1.place(x=10, y=160)
# 分类器训练模块
Button2 = tk.Button(root, text="读取图像库\n进行训练", font=('华文楷体', '20'), command=choose)
Button2.config(width=10, height=2)
Button2.place(x=10, y=260)
root.mainloop()
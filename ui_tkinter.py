import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from os import listdir
from os.path import isfile, join
from prediction import ImageClassifier
import shutil
from tkinter import messagebox



class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.store=[]
        

    def create_widgets(self):

        folderOptions = tk.LabelFrame(self, text="Folder Details: ")
        folderOptions.grid(row=0, columnspan=7, sticky='W', \
                 padx=5, pady=5, ipadx=5, ipady=5)

        #Folder with pictures
        inFileLbl = tk.Label(folderOptions, text="Folder with Pictures:") 
        inFileLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)

        inFileTxt = tk.Entry(folderOptions, width=80)
        inFileTxt.grid(row=0, column=1, columnspan=7, sticky="WE", pady=3)

        inFileBtn = tk.Button(folderOptions, text="Browse ...", command=self.directory_box(inFileTxt))
        inFileBtn.grid(row=0, column=8, sticky='W', padx=5, pady=2)

        #Destination Folder 1
        outFileLbl1 = tk.Label(folderOptions, text="Destination Folder 1:")
        outFileLbl1.grid(row=1, column=0, sticky='E', padx=5, pady=2)

        outFileTxt1 = tk.Entry(folderOptions, width=80)
        outFileTxt1.grid(row=1, column=1, columnspan=7, sticky="WE", pady=2)

        outFileBtn1 = tk.Button(folderOptions, text="Browse ...", command=self.directory_box(outFileTxt1))
        outFileBtn1.grid(row=1, column=8, sticky='W', padx=5, pady=2)

        #Destination Folder 2
        outFileLbl2 = tk.Label(folderOptions, text="Destination Folder 2:")
        outFileLbl2.grid(row=2, column=0, sticky='E', padx=5, pady=2)

        outFileTxt2 = tk.Entry(folderOptions, width=80)
        outFileTxt2.grid(row=2, column=1, columnspan=7, sticky="WE", pady=2)

        outFileBtn2 = tk.Button(folderOptions, text="Browse ...", command=self.directory_box(outFileTxt2))
        outFileBtn2.grid(row=2, column=8, sticky='W', padx=5, pady=2)

        sort = tk.Button(self, text="Process", bg="light green", width=50,command=self.make_operations)
        sort.grid(row=8, column=3, pady=6)


    def directory_box(self, widget,title=None, dirName=None): 
        def handler():
            options = {}
            options['initialdir'] = dirName
            options['title'] = title
            options['mustexist'] = False
            fileName = filedialog.askdirectory(**options)
            widget.delete('0', 'end')
            widget.insert('0', fileName)
            print("This is File name ",fileName)
            self.store.append(fileName)
            print(self.store)
        return handler

    def make_operations(self):

        """
        1.Access the images in the input directory.
        2.Predict the outcome of the images.
        3.Move the files to the respective directories based on the prediction outcomes.
        """
        #print("Source Folder",self.store[0])
        #print("Destination 1",self.store[1])
        #print("Destination 2",self.store[2])

        image_files = [join(self.store[0],filename) for filename in listdir(self.store[0])]
        #print(image_files)

        img_classifier = ImageClassifier(self.store[0])
        result = img_classifier.image_predict()

        for path,prediction in result.items():
            print(path,'-',prediction)
            if prediction == 'Upright':
                shutil.move(path,self.store[1])
            elif prediction == 'Sideways':
                shutil.move(path,self.store[2])

        messagebox.showinfo(title='Message',message='Successfully Sorted the Images..!!')

        root.destroy()

root = tk.Tk()
content = tk.Frame(root)
frame = tk.Frame(content, borderwidth=5, relief="sunken", width=200, height=100)

root.title("My Image Sorter")
app = Application(master=root)


app.mainloop()

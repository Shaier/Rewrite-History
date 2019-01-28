#Converting PDF files to txt files
import PyPDF2
import io
import os

for file in os.listdir('C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\Rewrite History'): #Where the pdf is located
    pdfFileObj = open('C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\Rewrite History\\'+file,'rb')     #'rb' for read binary mode
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    pdfReader.numPages #number of pages in the file
    with io.open('C:\\Users\\sagi\\Desktop\\Learning\\ML\\Datasets\\Rewrite History\\'+file+'.txt', 'w',encoding="utf-8") as f: #create a new txt file with the file's name
        for i in range (1,pdfReader.numPages):
            pageObj = pdfReader.getPage(i)          #'i' is the page number
            line=pageObj.extractText()
            f.write(line)

import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 200,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Calculate the Predicted Score:')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Enter the runs:')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry (root) 
canvas1.create_window(200, 150, window=entry1)


canvas2 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas2.pack()


label4 = tk.Label(root, text='Enter the Wickets:')
label4.config(font=('helvetica', 10))
canvas2.create_window(200, 10, window=label4)

entry2 = tk.Entry (root) 
canvas2.create_window(200, 50, window=entry2)

canvas3 = tk.Canvas(root, width = 400, height = 70,  relief = 'raised')
canvas3.pack()


label5 = tk.Label(root, text='Enter the Overs:')
label5.config(font=('helvetica', 10))
canvas3.create_window(200, 10, window=label5)

entry3 = tk.Entry (root) 
canvas3.create_window(200, 30, window=entry3)

canvas4 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas4.pack()


label6 = tk.Label(root, text='Enter the Striker runs:')
label6.config(font=('helvetica', 10))
canvas4.create_window(200, 10, window=label6)

entry4 = tk.Entry (root) 
canvas4.create_window(200, 30, window=entry4)

canvas5 = tk.Canvas(root, width = 400, height = 100,  relief = 'raised')
canvas5.pack()


label7 = tk.Label(root, text='Enter the Non-striker runs:')
label7.config(font=('helvetica', 10))
canvas5.create_window(200, 10, window=label7)

entry5 = tk.Entry (root) 
canvas5.create_window(200, 30, window=entry5)




def getSquareRoot ():
    
    x1 = entry1.get()
    x2=entry2.get()
    x3=entry3.get()
    x4=entry4.get()
    x5=entry5.get()
    
    label1 = tk.Label(root, text= float(x1)**0.5,font=('helvetica', 10, 'bold'))
    canvas5.create_window(200, 100, window=label1)
    
button1 = tk.Button(text='Get the Predicted score', command=getSquareRoot, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas5.create_window(200, 70, window=button1)

root.mainloop()
    
    
    
    
    

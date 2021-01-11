# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 05:37:24 2020
@author: Sanjay.N and Karanstan.J
Branch : B.Tech/Information Technology - III year 
"""
#importing basic and machine learning libraries
import time
import tkinter
import pandas as pd
from tkinter import ttk
import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#from sklearn.metrics import accuracy_score
#front end using tkinter module
root = tkinter.Tk()
root.title("ipl_prediction")
root.minsize(500,400) 
root.config(background="yellow")
root.title("IPL Predictor Machine")
def First():
    label=tkinter.Label(root,text='Welcome to IPL Predicter!!!',bg="yellow",fg="blue",font=("Lucida Calligraphy", 20))
    label.grid(column=0,row=1)
    ipl = "The Indian Premier League (IPL) is a professional Twenty20 cricket league in India \ncontested during March or April and May of every year by eight teams representing eight different cities in India. ...\n In 2010, the IPL became the first sporting event in the world to be broadcast live on YouTube."
    label2 = tkinter.Label(root,text=ipl,bg="yellow",fg = "blue",font=("Tw Cen MT Condensed Extra Bold",12))
    label2.grid(column=0,row=6)
    bt1=tkinter.Button(root,text='NaiveBayes',bg="blue",command=NaiveBayes)
    bt1.grid(column=0,row=7)
    bt2=tkinter.Button(root,text='RandomForest',bg="blue",command=RandomForestRegression)
    bt2.grid(column=0,row=8)
def NaiveBayes():
    root1 = tkinter.Tk()
    root1.minsize(800,500)
    root1.config(background="violet")
    root1.title("IPL Venue predicion")
    l = tkinter.Label(root1,text="Ipl Venue Prediction",bg="violet",fg = "purple",font=("Lucida Calligraphy", 20))
    l.grid(row=0,column=10, pady =15)
    l1 = tkinter.Label(root1, text = "MATCH HELD YEAR:",bg="violet",fg = "blue",font=("Lucida Calligraphy",14)) 
    l2 = tkinter.Label(root1, text = "TEAM-1:",bg="violet",fg = "blue",font=("Lucida Calligraphy",14)) 
    l3 = tkinter.Label(root1, text = "TEAM-2:",bg="violet",fg = "blue",font=("Lucida Calligraphy",14))
    l4 = tkinter.Label(root1, text = "TOSS-WINNER:",bg="violet",fg = "blue",font=("Lucida Calligraphy",14)) 
    l5 = tkinter.Label(root1, text = "TOSS-DECISION :",bg="violet",fg = "blue",font=("Lucida Calligraphy",14)) 
    l6 = tkinter.Label(root1, text = "WINNER:",bg="violet",fg = "blue",font=("Lucida Calligraphy",14))
    l1.grid(row = 1, column = 6, pady = 20) 
    l2.grid(row = 2, column = 6, pady = 20) 
    l3.grid(row = 3, column = 6, pady = 20) 
    l4.grid(row = 4, column = 6, pady = 20) 
    l5.grid(row = 5, column = 6, pady = 20) 
    l6.grid(row = 6, column = 6, pady = 20) 
    #logic
    def NaiveBayes():
        start_time = time.time()
        matches=pd.read_csv('E:\Projects\matches.csv')
        matches[pd.isnull(matches['winner'])]
        matches['winner'].fillna('Draw', inplace=True)
        matches['city'].fillna('Dubai',inplace=True)
        matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                     'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                     'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                    ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)
        matches.replace([2007,2008,2009,2010,2011,
                     2012,2013,2014,
                     2015,2016,2017]
                  ,['y7','y8','y9','y10','y11','y12','y13','y14','y15','y16','y17'],inplace=True)
    
        encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
              'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
              'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
              'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14},
             'toss_decision':{'field':1,'bat':2},
             'year':{'y7':1,'y8':2,'y9':3,'y10':4,'y11':5,'y12':6,'y13':7,'y14':8,'y15':9,'y16':10,'y17':11}}
        matches.replace(encode, inplace=True)
        X_train = pd.get_dummies(matches[['year','team1', 'team2', 'toss_winner','toss_decision','winner']])
        y_train = pd.DataFrame(matches['venue'])
        #Create a Gaussian Classifier
        model = GaussianNB()
        # Train the model using the training sets 
        model.fit(X_train, y_train)
        #Predict Output 
        a ={'2007':1,'2008':2,'2009':3,'2010':4,'2011':5,'2012':6,'2013':7,'2014':8,'2015':9,'2016':10,'2017':11}[year.get()] 
        b ={'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13}[team1.get()] 
        c ={'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13}[team2.get()]
        d ={'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13}[toss_winner.get()] 
        e ={'field':1,'bat':2}[toss_desicion.get()] 
        f ={'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13}[winner.get()]
        pred= model.predict([[a,b,c,d,e,f]])
        pred = str(pred)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=1)
        y_pred = model.predict(X_test)
        accurcy = metrics.accuracy_score(y_test,y_pred)*100
    
        end_time = time.time()-start_time
        print(sns.countplot(x="winner",data=matches))
        
        l7 = tkinter.Label(root1,text = "Match to be held at:",bg="violet", fg = "green",font=("Engravers MT", 12))
        l7.grid(row = 7, column = 7, pady = 5) 
        l8 = tkinter.Label(root1, text = pred,bg="violet", fg = "green",font=("Engravers MT", 12))
        l8.grid(row = 7, column = 8, pady = 5)  
        l9 = tkinter.Label(root1,text = "Time taken to predict the result in (seconds):",bg="violet",fg = "green",font=("Engravers MT",12))
        l9.grid(row = 8, column = 7, pady = 20) 
        l10 = tkinter.Label(root1,text = end_time,bg="violet",fg = "green",font=("Engravers MT", 12))
        l10.grid(row = 8, column = 8, pady = 20) 
        l11 = tkinter.Label(root1,text = "Accuracy of constructed model:",bg="violet",fg = "green",font=("Engravers MT", 12))
        l11.grid(row = 9, column = 7, pady = 20) 
        l12 = tkinter.Label(root1,text =accurcy,fg = "green",bg="violet",font=("Engravers MT", 12) )
        l12.grid(row = 9, column = 8, pady = 20) 
        
    #creating combo box
    n1 = tkinter.StringVar()
    year = ttk.Combobox(root1, width = 27, textvariable = n1) 
    year['values'] = ('2007','2008','2009','2010','2011','2012','2013','2014','2015','2017') 
    year.grid(column = 8, row = 1) 
    year.current(1)
    
    n2 = tkinter.StringVar()
    team1 = ttk.Combobox(root1, width = 27, textvariable = n2)  
    team1['values'] = ('MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW') 
    team1.grid(column = 8, row = 2) 
    team1.current(1)
    
    n3 = tkinter.StringVar()
    team2 = ttk.Combobox(root1, width = 27, textvariable = n3)  
    team2['values'] = ('MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW') 
    team2.grid(column = 8, row = 3) 
    team2.current(1)
    
    n4 = tkinter.StringVar()
    toss_winner = ttk.Combobox(root1, width = 27, textvariable = n4)  
    toss_winner['values'] = ('MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW') 
    toss_winner.grid(column = 8, row = 4) 
    toss_winner.current(1)
    
    n5 = tkinter.StringVar()
    toss_desicion = ttk.Combobox(root1, width = 27, textvariable = n5)  
    toss_desicion['values'] = ('field','bat') 
    toss_desicion.grid(column = 8, row = 5) 
    toss_desicion.current(1)
    
    n6 = tkinter.StringVar()
    winner = ttk.Combobox(root1, width = 27, textvariable = n6)  
    winner['values'] = ('MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW') 
    winner.grid(column = 8, row = 6) 
    winner.current(1)
    
    
    sub = tkinter.Button(root1, text = 'submit',command=NaiveBayes, bd = '5')
    #sub['command'] = NaiveBayes()
    sub.grid(row=10,column = 10,pady = 20)
    root1.mainloop()
def RandomForestRegression():
    root2 = tkinter.Tk()
    root2.minsize(800,500)
    root2.config(background="violet")
    root2.title("IPL Venue predicion(RandomForestRegression)")
    start_time = time.time()
    matches=pd.read_csv('E:\Projects\matches.csv')
    matches['winner'].fillna('Draw', inplace=True)
    matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                     'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                     'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                    ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)
    
    encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
              'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
              'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
              'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
    matches.replace(encode, inplace=True)
    matches['city'].fillna('Dubai',inplace=True)
    matches = matches[['team1','team2','year','toss_decision','toss_winner','venue','winner']]
    df = pd.DataFrame(matches)
    
    from sklearn.preprocessing import LabelEncoder
    var_mod = ['year','toss_decision','venue']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    df.dtypes 
    def classification_model(model, data, predictors, outcome):
      model.fit(data[predictors],data[outcome])
      predictions = model.predict(data[predictors])
      accuracy = metrics.accuracy_score(predictions,data[outcome])
      l12 = tkinter.Label(root2,text =accuracy,fg = "green",bg="violet",font=("Engravers MT", 12) )
      l12.grid(row = 2, column = 8, pady = 20) 
    outcome_var=['venue']
    predictor_var = ['team1', 'team2','toss_winner','year','winner','toss_decision']
    model = LogisticRegression()
    classification_model(model, df,predictor_var,outcome_var)
    end_time = time.time()-start_time
    l9 = tkinter.Label(root2,text = "Time taken to predict the result in (seconds):",bg="violet",fg = "green",font=("Engravers MT",12))
    l9.grid(row = 1, column = 7, pady = 20) 
    l10 = tkinter.Label(root2,text = end_time,bg="violet",fg = "green",font=("Engravers MT", 12))
    l10.grid(row = 1, column = 8, pady = 20) 
    l11 = tkinter.Label(root2,text = "Accuracy of constructed model:",bg="violet",fg = "green",font=("Engravers MT", 12))
    l11.grid(row = 2, column = 7, pady = 20)  
First()
root.mainloop()
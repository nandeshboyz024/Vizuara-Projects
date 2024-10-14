from flask import Flask, request, jsonify
app = Flask(__name__)
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

global mydata
mydata=[]
global rf_model


def dataframe_difference(df1, df2, which=None):
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df.text=df.text.apply(lambda text : lower_case(text))
    df.text=df.text.apply(lambda text : Removing_numbers(text))
    df.text=df.text.apply(lambda text : Removing_punctuations(text))
    df.text=df.text.apply(lambda text : Removing_urls(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    return sentence

def train_model(model,data,targets):
    # Term Frequency-Inverse Document Frequency (TF-IDF)
    text_clf=Pipeline([('vect',TfidfVectorizer()),('clf',model)])
    text_clf.fit(data,targets)
    return text_clf


@app.route('/train-model',methods=['POST'])
def trainModel():
     try:
          global mydata
          train=pd.DataFrame(mydata,columns=['text','emotion'])
          train=normalize_text(train)
          X_train=train['text'].values
          y_train=train['emotion'].values
          global rf_model
          rf_model=train_model(RandomForestClassifier(random_state=0),X_train,y_train)
          return {'success':True}
     except:
          return {'success':False}



@app.route('/predict-emotion',methods=['POST'])
def predict_emotion():
    try:
         chances = [0,0,0,0,0,0]
         data = request.get_json()
         query = data['query']
         global rf_model
         predict_arr=rf_model.predict([normalized_sentence(query)])
         predict_proba_arr=rf_model.predict_proba([normalized_sentence(query)])
         pred=predict_arr[0]
         for emotion,probability in zip(rf_model.classes_,predict_proba_arr[0]):
            if(emotion=='joy'):
                chances[0]=round(probability*100)
            elif(emotion=='sadness'):
                chances[1]=round(probability*100)
            elif(emotion=='anger'):
                chances[2]=round(probability*100)
            elif(emotion=='fear'):
                chances[3]=round(probability*100)
            elif(emotion=='love'):
                chances[4]=round(probability*100)
            elif(emotion=='surprise'):
                chances[5]=round(probability*100)
         response={
              'prediction':pred,
              'chances':chances
		 }   
         print(pred,chances)
         return jsonify(response)
    except Exception as e:
         return jsonify({'error':str(e)})


@app.route('/accessData')
def accessData():
    global mydata
    response={
        'data':mydata
    }
    return jsonify(response)

@app.route('/data-size')
def accessDataSize():
    global mydata
    response={
        'dataSize':int(len(mydata))
    }
    return jsonify(response)

@app.route('/add-data',methods=['POST'])
def addData():
     try:
          global mydata
          newData=request.get_json()
          text=newData['text']
          emotion=newData['emotion']
          mydata.append([text,emotion])
          return jsonify({'success':True})
     except:
          return jsonify({'success':False,'error':'nvalid Request'})

@app.route('/pop',methods=['POST'])
def popData():
     try:
          global mydata
          mydata.pop()
          return jsonify({'success':True})
     except:
          return jsonify({'success':False,'error':'Invalid request'})

@app.route('/reset',methods=['POST'])
def resetData():
     try:
          global mydata
          mydata.clear()
          return jsonify({'success':True})
     except:
          return jsonify({'success':False,'error':'Invalid request'})          

@app.route('/get-size')
def getSize():
          global mydata
          train=pd.DataFrame(mydata,columns=['text','emotion'])
          size=[
               int(train.emotion.value_counts().get('joy',0)),
               int(train.emotion.value_counts().get('sadness',0)),
               int(train.emotion.value_counts().get('anger',0)),
               int(train.emotion.value_counts().get('fear',0)),
               int(train.emotion.value_counts().get('love',0)),
               int(train.emotion.value_counts().get('surprise',0))
            ]
          response={
               'count':size,
          }
          return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

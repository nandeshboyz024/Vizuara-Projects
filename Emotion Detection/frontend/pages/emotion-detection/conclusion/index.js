import React from 'react'
import Head from 'next/head'
import Sidebar from '../../../components/emotion-detection/Sidebar'
export default function Conclusion() {
  return (
    <div>
        <Head>
          <title>
            Emotion Detection : Conclusion
          </title>
        </Head>
      <div className="container-fluid">
        <div className="row">
          <div className="col-sm-3 ">
            <Sidebar/>
          </div>
          <div className="col-sm-9">
            <div style={{textAlign:'center',fontFamily:'Helvetica Neue',marginTop:'10px'}}>
              <h1 className="heading" style={{background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip:'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Conclusion</h1>
            </div>
            <div style={{fontFamily:'Helvetica Neue',textAlign:'justify',fontSize:'20px'}}>
                <p><b>Words have Feelings: </b> Just like people, words can express different emotions. For example, &quot;happy&quot;, &quot;joyful&quot;, and &quot;excited&quot; all show positive feelings, while &quot;sad&quot;, &quot;disappointed&quot;, and &quot;lonely&quot; express sadness.</p>
                <p><b>Detecting Emotions: </b> Computers are super smart at understanding patterns. Scientists and computer experts have taught computers to recognize these emotional patterns in the words we use. So, when you type, the computer can guess how you are feeling!</p>
                <p><b>Emoji Friends:</b> You know those cute emojis you use in your messages? They are like little helpers for computers. Emojis can give extra clues about emotions. 😃 means happy, 😢 means sad, and there are emojis for all sorts of feelings!</p>
                <p><b>Helping Us:</b> Emotion recognition can be useful! For example, it helps to analyze customer reviews in business, monitoring patient feedback tn healthcare services, tweets in social media platform.</p>
                So, the next time you send a message or write a story, remember that computers can understand your emotions too! It is like having a friend who can read between the lines and know exactly how you feel.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

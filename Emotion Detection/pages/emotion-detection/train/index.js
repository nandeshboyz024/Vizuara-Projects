import React,{useState} from 'react'
import Sidebar from '../../../components/emotion-detection/Sidebar'
import {useEmotionContext} from '../../../context/EmotionContext'
import Head from 'next/head';
import Image from 'next/image';
export default function Train() {
    const {myData,setIsTrain} = useEmotionContext();
    const [alert,setAlert]=useState(null);
    const showAlert = (message,type)=>{
        setAlert({
            msg:message,
            type:type
        })
        setTimeout(()=>{
            setAlert(null);
        },2000);
    };
    const trainModel=async()=>{
            
            try{
                const response = await fetch('https://coral-app-kcz3j.ondigitalocean.app/train-model',{
                    method:'POST',
                    headers:{
                        'Content-Type':'application/json',
                    },
                    body:JSON.stringify(myData),
                });
                
                const result = await response.json();
                
                if(result.success){
                    setIsTrain(true);
                    showAlert("The model has been succeessfully trained.","success");
                }
                else{
                    
                    showAlert("something went wrong! Please try again","warning");
                }
            }
            catch(error){
                console.error('Error during fetch:', error);
                showAlert('Failed to train model','warning');
            }
    }
  return (
    <div>
        <Head>
            <title>
                Emotion Detection : Train Model
            </title>
        </Head>
        <div className="container-fluid">
            <div className="row">
                <div className="col-sm-3 ">
                    <Sidebar/>
                </div>
                <div className="col-sm-9">
                    {alert && (
                        <div className={`alert alert-${alert.type} `} role="alert">
                            <strong>{alert.type}</strong>: {alert.msg}
                        </div>
                    )}
                    <div style ={{textAlign:'center'}}>
                        <h1 className="heading" style={{textAlign:'center',background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip:'text',color: 'transparent',fontFamily:'Helvetica Neue',display:'inline'}}>Train-Model</h1>
                    </div>
                    <div className="container" style={{marginTop:'10px'}}>
                        <div className="row">
                            <div className="col-lg-6" style={{fontFamily:'Helvetica Neue',textAlign:'justify',fontSize:'20px'}}>                                
                                <p>This model is like a child. Whatever you teach it, it will learn. The more you teach it, the better it will learn, and it will tell you as well as it has learned when you ask.</p>
                                <p>If It makes incorrect predictions, train it and teach it what is right. In other words, add new data to the training data.</p>
                            </div>
                            <div className="col-lg-6">
                                <Image src="/images/bert.png" style={{width:'400px'}} alt="" srcSet=""/>
                            </div>
                        </div>
                    </div>
                    <hr/>
                    { myData && myData.length>0 ? (
                        <div style={{textAlign:'center'}}>
                            <button className="btn_style" onClick={trainModel}>Train-Model</button>
                        </div>
                        ) :
                        (<div style={{textAlign:'center',fontFamily:'Helvetica Neue'}}>
                            <p style={{color:'red'}}> <i>For Train the Model, Your dataset should not be empty.</i></p>
                        </div>)
                    }
                </div>
            </div>
        </div>
    </div>
  )
}

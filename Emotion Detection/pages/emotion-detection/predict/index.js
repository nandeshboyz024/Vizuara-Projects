import React, {useState,useEffect} from 'react'
import Sidebar from '../../../components/emotion-detection/Sidebar'
import {useEmotionContext} from '../../../context/EmotionContext'
import Head from 'next/head';
import Image from 'next/image';
export default function Predict() {
    const {isTrain} = useEmotionContext();
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
    const [text,setText]=useState("");
    const [prediction,setPrediction]=useState(null);
    const [song,setSong]=useState(null);
    const [sticker,setSticker]=useState("/media/welcome.gif");
    const songs={
        joy:"/media/joy.mp3",
        sadness:"/media/sadness.mp3",
        anger:"/media/anger.mp3",
        fear:"/media/fear.mp3",
        love:"/media/love.mp3",
        surprise:"/media/surprise.mp3",
    };
    const stickers={
        joy:"/media/joy.gif",
        sadness:"/media/sadness.gif",
        anger:"/media/anger.gif",
        fear:"/media/fear.gif",
        love:"/media/love.gif",
        surprise:"/media/surprise.gif",
    };
    const handleTextOnChange=(event)=>{
        setText(event.target.value);
    }
    const handleSubmit = async(e) => {
        e.preventDefault();
        const toggleButton = document.getElementById("toggleButton");
        if(!song.paused){
            song.pause();
            toggleButton.textContent = "Play Music";
        }
        if(!text.trim()){
            showAlert("Please provide some text","warning");
            return;
        }
        if(! isTrain){
            showAlert("Please train Your Model First","warning");
            return;
        }
        // Add your logic to store or process the form data here
        try{
            const response = await fetch("https://coral-app-kcz3j.ondigitalocean.app/predict-emotion",{
                method:'POST',
                headers:{
                    'Content-Type':'application/json',
                },
                body:JSON.stringify({'query':text}),
            });
            
            const responseData = await response.json();
            //console.log(responseData);
            setPrediction(responseData);
            //console.log("dekh "+responseData.prediction);
            //console.log(songs[prediction.prediction]);
            setSong(new Audio(songs[responseData.prediction]));
            setSticker(stickers[responseData.prediction]);
        }
        catch (error){
            console.error('Error submitting the form:',error);
        }
    };
    const toggleMusic = ()=>{
        const toggleButton = document.getElementById("toggleButton");
        if (song.paused) {
            song.play();
            toggleButton.textContent = "Pause Music";
        }
        else {
            song.pause();
            toggleButton.textContent = "Play Music";
        }
    }

    useEffect(() => {
        if (song) {
          const handleSongEnd = () => {
            const toggleButton = document.getElementById('toggleButton');
            toggleButton.textContent = 'Play Music';
          };
      
          song.addEventListener('ended', handleSongEnd);
      
          return () => {
            song.removeEventListener('ended', handleSongEnd);
          };
        }
      }, [song]);
      
    useEffect(()=>{
        setSong(new Audio('/media/bgm.mp3'));
    },[])

  return (
    <div>
        <Head>
            <title>
                Emotion Detection : Predict
            </title>
        </Head>
        <div className="container-fluid">
            <div className="row">
                <div className="col-sm-3 ">
                    <Sidebar/>
                </div>
                <div className="col-sm-9">
                    <div>
                        {alert && (
                            <div className={`alert alert-${alert.type} `} role="alert">
                                <strong>{alert.type}</strong>: {alert.msg}
                            </div>
                        )}
                    </div>
                    <div style ={{textAlign:'center'}}>
                        <h1 className="heading" style={{textAlign:'center',background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip: 'text',color: 'transparent',fontFamily:'Helvetica Neue',display:'inline'}}>Prediction</h1>
                    </div>
                    <div style={{margin:'10px',textAlign:'right'}}>
                        <button id="toggleButton" onClick={toggleMusic}>Play Music</button>
                    </div>
                    <div className="container">
                        <div className = "row">
                            <div className="col-lg-8">
                                <div className="card">
                                    <div style={{textAlign:'center', marginTop:'10px',marginBottom:'10px'}}>
                                        <h2 style={{fontWeight:'bold',background: 'linear-gradient(to right, #FCC978, #FB5EA3)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Ask Your</h2>
                                        <h2 style={{fontWeight:'bold',background: 'linear-gradient(to right, #FB5CA6, #FA00FF)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Query</h2>
                                        <div>
                                            <form onSubmit = {handleSubmit}>
                                                <div>
                                                    <textarea  id="myText" style={{height: '100px',width:'80%'}} value={text} onChange={handleTextOnChange}></textarea>
                                                </div>
                                                    <input id="predict" type="submit" value="Predict"/>
                                            </form>
                                        </div>
                                    </div>
                                </div>
                                <div className="container">
                                    <h3>Chance of each emotion</h3>
                                    <div class ="row">
                                        <div className="col-3">
                                            <h6>joy</h6>
                                        </div>
                                        <div className="col-8">
                                            <div className="progress">
                                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style={{width:`${prediction &&  prediction.chances[0]}%`,backgroundColor: '#39e75f',color:'black'}} area-valuenow={prediction &&  prediction.chances[0]} aria-valuemin="0" area-valuemax="100">
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-1" style={{fontSize:'13px',paddingLeft:'0',marginTop:'-2px'}}>{prediction &&  `${prediction.chances[0]}%`}</div>
                                    </div>
                                    <div class ="row">
                                        <div className="col-3">
                                            <h6>sadness</h6>
                                        </div>
                                        <div className="col-8">
                                            <div className="progress">
                                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style={{width:`${prediction &&  prediction.chances[1]}%`,backgroundColor: 'blue',opacity: '0.8',color:'black'}} area-valuenow={prediction &&  prediction.chances[1]} aria-valuemin="0" area-valuemax="100">
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-1" style={{fontSize:'13px',paddingLeft:'0',marginTop:'-2px'}}>{prediction &&  `${prediction.chances[1]}%`}</div>
                                    </div>
                                    <div class ="row">
                                        <div className="col-3">
                                            <h6>anger</h6>
                                        </div>
                                        <div className="col-8">
                                            <div className="progress">
                                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style={{width:`${prediction &&  prediction.chances[2]}%`,backgroundColor: 'red',opacity: '0.8',color:'black'}} area-valuenow={prediction &&  prediction.chances[2]} aria-valuemin="0" area-valuemax="100">
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-1" style={{fontSize:'13px',paddingLeft:'0',marginTop:'-2px'}}>{prediction &&  `${prediction.chances[2]}%`}</div>
                                    </div>
                                    <div class ="row">
                                        <div className="col-3">
                                            <h6>fear</h6>
                                        </div>
                                        <div className="col-8">
                                            <div className="progress">
                                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style={{width:`${prediction &&  prediction.chances[3]}%`,backgroundColor: 'green',opacity: '0.8',color:'black'}} area-valuenow={prediction &&  prediction.chances[3]} aria-valuemin="0" area-valuemax="100">
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-1" style={{fontSize:'13px',paddingLeft:'0',marginTop:'-2px'}}>{prediction &&  `${prediction.chances[3]}%`}</div>
                                    </div>
                                    <div class ="row">
                                        <div className="col-3">
                                            <h6>love</h6>
                                        </div>
                                        <div className="col-8">
                                            <div className="progress">
                                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style={{width:`${prediction &&  prediction.chances[4]}%`,backgroundColor: '#fc0beb',opacity: '0.8',color:'black'}} area-valuenow={prediction &&  prediction.chances[4]} aria-valuemin="0" area-valuemax="100">
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-1" style={{fontSize:'13px',paddingLeft:'0',marginTop:'-2px'}}>{prediction &&  `${prediction.chances[4]}%`}</div>
                                    </div>
                                    <div class ="row">
                                        <div className="col-3">
                                            <h6>surprise</h6>
                                        </div>
                                        <div className="col-8">
                                            <div className="progress">
                                                <div className="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style={{width:`${prediction &&  prediction.chances[5]}%`,backgroundColor: 'gold',opacity: '0.8',color:'black'}} area-valuenow={prediction &&  prediction.chances[5]} aria-valuemin="0" area-valuemax="100">
                                                </div>
                                            </div>
                                        </div>
                                        <div className="col-1" style={{fontSize:'13px',paddingLeft:'0',marginTop:'-2px'}}>{prediction &&  `${prediction.chances[5]}%`}</div>
                                    </div>
                                </div>
                                
                            </div>
                            <div className="col-lg-4">
                                <div className="card" style={{textAlign:'center'}}>
                                    <h2 id="song_id" style={{fontWeight:'bold',background: 'linear-gradient(to right, #FCC978, #FB5EA3)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}} >My Prediction</h2>
                                    <h2 style={{fontWeight:'bold',background: 'linear-gradient(to right, #FB5CA6, #FA00FF)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>{prediction &&  prediction.prediction}</h2>
                                    <Image src={sticker} alt="predicted emotion" srcSet=""/>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
  )
}

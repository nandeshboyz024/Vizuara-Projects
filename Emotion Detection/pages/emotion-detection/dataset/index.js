import React, { useEffect, useState }  from 'react'
import Sidebar from '../../../components/emotion-detection/Sidebar'
import Animation from '../../../components/emotion-detection/Animation'
import {useEmotionContext} from '../../../context/EmotionContext'
import Head from 'next/head'
export default function Dataset() {
    const {myData, setMyData, size, setSize } = useEmotionContext();
    const [alert,setAlert]=useState(null);
    const [emotion,setEmotion]=useState("6");
    const [text,setText]=useState("");
    const emotions={
        "0":'joy',
        "1":'sadness',
        "2":'anger',
        "3":'fear',
        "4":'love',
        "5":'surprise',
    }
    const showAlert=(message,type)=>{
        setAlert({
            message:message,
            type:type,
        })
        setTimeout(() => {
            setAlert(null);
        }, 2000);
    }  
    const deleteLastEntry = ()=>{
        size[myData[myData.length-1][1]]--;
        const newData = myData;
        newData.pop();
        setMyData(newData);    
        showAlert('Last entry deleted successfully','success');
    }
    const resetData=()=>{
        if(myData.length === 0) {
            showAlert("Your dataset is already empty","info");
            return;
        }
        const wish = window.confirm("Do you want to reset your dataset ?");
        if(wish){
            setMyData([]);
            setSize({
                'joy':0,
                'sadness':0,
                'anger':0,
                'fear':0,
                'love':0,
                'surprise':0
            });
            showAlert("Your dataset is reset successfully","success");
        }
    }
    const toggleData=()=>{
        const btn = document.getElementById('showmy');
        if(btn.textContent ==="Show My Data"){
            if(myData.length===0){
                showAlert("You have no data in your dataset.",'info');
               return;
            }
            btn.textContent="Hide My Data";
        }
        else{
            btn.textContent="Show My Data";
        }
        const myDiv = document.getElementById('my_data');
        myDiv.classList.toggle('visible');
    }
    const handleTextOnChange=(event)=>{
       setText(event.target.value);
    }
    const handleEmotionOnChange=(event)=>{
        setEmotion(event.target.value);
    }
    const handleSubmit = async(event) => {
        event.preventDefault();
        if(!text.trim()){
            showAlert("Please provide some text","warning");
            return;
        }
        if(emotion==="6"){
            showAlert("Please select emotion-class","warning");
            return;
        }
        const newData = myData;
        newData.push([text,emotions[emotion]]);
        setMyData(newData);
        size[emotions[emotion]]++;
        showAlert("your data is added successfully","success");
        setText("");
        setEmotion("6");
    }
    useEffect(() => {
        Animation();
      }, []);
  return (
    <div>
        <Head>
            <title>
                Emotion Detection : Dataset
            </title>
        </Head>
        <div className="container-fluid">
            <div className="row">
                <div className="col-sm-3 ">
                    <Sidebar/>
                </div>
                <div className="col-sm-9">
                    <div style={{textAlign:'center',fontFamily:'Helvetica Neue',marginTop:'10px'}}>
                        <h1 className="heading" style={{background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Create Your Own Dataset</h1>
                    </div>
                    
                    <div className="card" id = "mydiv" style={{marginTop:'20px',paddingBottom:'10px',position: 'relative',zIndex:'1'}}>  
                        <canvas id="cv" style={{position:'absolute',zIndex:'-1'}}></canvas>
                        <form onSubmit = {handleSubmit}>
                            <div className="container">
                                <div className="row" style={{marginLeft:'5px'}}>
                                    <div className="col-lg-4" style={{textAlign:'center'}} >
                                        <h3 style={{background: 'linear-gradient(to right,  #FCC978, #FB5EA3)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Select</h3>
                                        <h3 style={{background: 'linear-gradient(to right, #FB5CA6, #FA00FF)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Emotion-Class</h3>
                                        <select name="emotion" id="emotion" value={emotion} onChange={handleEmotionOnChange}>
                                            <option value="6" >Select...</option>
                                            <option value="0">joy</option>
                                            <option value="1">sadness</option>
                                            <option value="2">anger</option>
                                            <option value="3">fear</option>
                                            <option value="4">love</option>
                                            <option value="5">surprise</option>
                                        </select>
                                    </div>
                                    <div className="col-lg-8" style={{textAlign:'center'}}>
                                        <h3 style={{background: 'linear-gradient(to right,  #FCC978, #FB5EA3)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Write Here...</h3>
                                        <textarea  id="myText" style={{height: '100px',width:'90%'}} value={text} onChange={handleTextOnChange}></textarea>
                                        <input className="btn_style" id="add_data" type="submit" value="Add Data"/>
                                    </div>
                                </div>
                            </div>
                        </form>
                    </div>
                    
                    <div style={{marginTop:'20px',display: 'flex', justifyContent: 'space-between',alignItems: 'center'}}>
                        <button className="btn_style" id="showmy" onClick={toggleData}>Show My Data</button>
                        <button className="btn_style" id="resetmy" onClick={resetData}>Reset My Data</button>
                    </div>
                    <div>
                        {alert && (
                            <div className={`alert alert-${alert.type} `} role="alert">
                                <strong>{alert.type}</strong>: {alert.message}
                            </div>
                        )}
                    </div>
                    <div id="my_data" className="hidden" style={{marginTop:'15px'}}>
                        { myData && myData.length>0 &&
                            (
                                <div>
                                    <div style={{textAlign:'center'}}>
                                        <button className="btn_style" id="popmy" onClick={deleteLastEntry} >Delete Your Last Entry</button>
                                    </div>
                                    <div className="my_data_cls" style={{marginTop:'15px'}}>
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Text</th>
                                                    <th>Label</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                { 
                                                myData.slice().reverse().map((entry,index)=>(
                                                        <tr key={index}>
                                                            <td>{entry[0]}</td>
                                                            <td>{entry[1]}</td>
                                                        </tr>
                                                    )
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>    
                            )
                        }
                    </div>
                </div>
            </div>
        </div>
    </div>    
  )
}

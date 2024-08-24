import React,{useState} from 'react'
import Sidebar from '../components/emotion-detection/Sidebar'
import Image from 'next/image'
export default function Home() {
  return (
    <div>
        <div className="container-fluid">
            <div className="row">
                <div className="col-sm-3 ">
                    <Sidebar/>
                </div>
                <div className="col-sm-9">
                    <div style={{textAlign:'center',fontFamily:'Helvetica Neue'}}>
                        <h1 className="heading" style={{background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip:'text',color: 'transparent',fontFamily:'Helvetica Neue'}}>AI Student Emotion Detector</h1>
                    </div>
                    <div className="container" style={{marginTop:'10px'}}>
                        <div className="row">
                            <div className="col-lg-5" style={{textAlign:'justify',fontFamily:'Helvetica Neue',fontSize:'20px'}}>
                                <p>Hello there! Have you ever wondered how computers can understand our feelings, just by reading what we write? It is like magic, but it is actually a cool thing called <b>Emotion Recognition using Text</b>!</p>
                                <p>Imagine you are texting your friend about your day. You might say, &quot;I had so much fun at the park!&quot; or &quot;I felt a bit sad because it rained during our picnic.&quot; Well, computers can read these messages and figure out if you are happy, sad, or maybe even excited!</p>   
                            </div>
                            <div className="col-lg-1"> </div>
                            <div className="col-lg-6">
                                <Image src="images/img_emotion1.png" style={{width:'450px'}} alt="" srcSet=""/>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
  )
}

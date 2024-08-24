import React from 'react'
import BarChart from '../../../components/emotion-detection/BarChart'
import PieChart from '../../../components/emotion-detection/PieChart'
import Sidebar from '../../../components/emotion-detection/Sidebar'
import {useEmotionContext} from '../../../context/EmotionContext'
import Head from 'next/head'
export default function Visualization() {
    const {size} = useEmotionContext();
    const data_size =[
      size && size.joy||0,
      size && size.sadness||0,
      size && size.anger||0,
      size && size.fear||0,
      size && size.love||0,
      size && size.surprise||0,
    ];
  return (
    <div>
        <Head>
            <title>
                Emotion Detection : Visualization
            </title>
        </Head>
        <div className="container-fluid">
            <div className="row">
                <div className="col-sm-3 ">
                    <Sidebar/>
                </div>
                <div className="col-sm-9">
                    <div style={{textAlign:'center',fontFamily:'Helvetica Neue',marginTop:'10px'}}>
                        <h1 className="heading" style={{background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip:'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Data-Visualization</h1>
                    </div>
                    <div className='container'>
                        <div className='row'>
                            <div className='col-lg-6'>
                                <BarChart chartId="myChart1" data={data_size} />
                            </div>
                            <div className='col-lg-6'>
                                <PieChart chartId="myChart2" data={data_size}/>
                            </div>
                        </div>
                    </div>
                </div>
            </div> 
        </div>       
    </div>
  )
}

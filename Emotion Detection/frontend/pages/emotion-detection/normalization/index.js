import React from 'react'
import Sidebar from '../../../components/emotion-detection/Sidebar'
import Head from 'next/head'
export default function Normalization() {
  return (
    <div>
        <Head>
          <title>
            Emotion Detection : Normalization
          </title>
        </Head>
        <div className="container-fluid">
            <div className="row">
                <div className="col-sm-3 ">
                    <Sidebar/>
                </div>
                <div className="col-sm-9">
                  <div style={{textAlign:'center',fontFamily:'Helvetica Neue',marginTop:'10px'}}>
                      <h1 className="heading" style={{background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)', WebkitBackgroundClip: 'text', color: 'transparent',fontFamily:'Helvetica Neue'}}>Data Normalization</h1>
                  </div>
                  <div style={{fontFamily:'Helvetica Neue',textAlign:'justify',fontSize:'20px'}}>
                      <h3>How and Why?</h3>
                      <ul>
                          <li>Normalisation helps us to minimize data redundancy by removing duplicate data beacuse it may be cause of bias.</li>
                          <li>Normalisation ensures consistency of data across database. Without normalization, if data is duplicated across multiple records and one copy is updated or deleted, it is easy to miss updating or deleting the corresponfing data in other places.</li>
                          <li>Data should not be misclassified otherwise it can be cuase of wrong prediction. For example, if you add text &quot;I am happy&quot; in sadness class, then model will predict it as sadness which is not good to our purpose.</li>
                          <li>Data should be according to feature of the dataset. It is meaningless to have mobile no. or website link in text like &quot;I received a missed call from +919887961384.&quot; or &quot;the site of Google, &apos; https://www.google.com/ &apos;, is not opening on my mobile.&quot; </li>
                      </ul>
                  </div>
                </div>
              </div>
            </div>  
    </div>
  )
}

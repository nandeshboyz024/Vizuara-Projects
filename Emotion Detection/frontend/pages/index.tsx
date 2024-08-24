import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import Head from 'next/head';
const Index = () => {
  const [isBrowser, setIsBrowser] = useState(false);
  useEffect(() => {
    setIsBrowser(true);
  }, []);

  return (
    <div style={{background:'radial-gradient(circle at 10% 20%, rgb(255, 12, 253) 0%, rgb(255, 241, 0) 80%)',margin:'0',padding:'0'}}>
      <Head>
        <title>
          Emotion Detection using Text
        </title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
   <div className='text-center' style={{padding:'10px'}}>
    <h1>Emotion Detection using Text</h1>
    <p style={{fontSize:'18px'}}>
      Emotion detection, also known as sentiment analysis, is a fascinating field within natural language processing (NLP) that focuses on understanding and interpreting the emotions expressed in textual content. In the context of this project, we delve into the intriguing realm of Emotion Detection using Text.
    </p>
    <p style={{fontSize:'18px'}}>
      As humans, we naturally convey our feelings, opinions, and sentiments through the words we use. Whether it is a social media post, customer reviews, or news articles, text is a rich source of emotional expression. Emotion detection algorithms enable us to analyze and comprehend these sentiments programmatically, providing valuable insights into the emotional tone and mood behind the words.
    </p>
    <p style={{fontSize:'18px'}}>
      This project aims to explore and implement methods for detecting emotions in textual data. By leveraging advanced machine learning techniques and natural language processing models, we strive to build a system that can automatically recognize and categorize emotions such as joy, sadness, anger, and more.
    </p>
    <p style={{fontSize:'18px'}}>
      Welcome to our journey into Emotion Detection using Text! Imagine teaching computers to understand feelings - that is exactly what we are up to. 
    </p>
    <button className='btn_style' style={{padding:'5px'}}><Link href='/emotion-detection/' className="nav-link ">Let&apos;s Start</Link></button>   
   </div>
   </div>
  );
};

export default Index;

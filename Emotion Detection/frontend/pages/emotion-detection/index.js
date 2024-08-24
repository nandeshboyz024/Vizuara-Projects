import React from 'react'
import Home from '../home'
import Head from 'next/head'
import {EmotionContextProvider} from '../../context/EmotionContext'
export default function index() {
  return (
    <div>
        <Head>
          <title>
            Emotion Detection Using Text
          </title>
        </Head>
        <Home/>
    </div>
  )
}

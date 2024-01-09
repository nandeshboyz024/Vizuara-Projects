import React from 'react'
import styles from '../../../components/animal-detection/classic.module.css'
import Heading from '../../../components/animal-detection/heading'
import {moveNext,movePrev,allowDrop,drag,drop} from '../../../components/animal-detection/function'
import Link from 'next/link'
import Image from 'next/image'
export default function CatClass() {

  return (
    <div>
        <Heading/>
        <div>
            <div style={{display:'flex'}}>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className={styles.arrow} id="prev" onClick={movePrev}>
                        <Image src="/imgs/left-arrow.png" alt="left-arrow"  width={20} height={20}/>
                    </button>
                </div>
                <div style={{width:'90%'}}>
                    <div className={styles.container} id="c1" style={{textAlign:'center'}}>
                        <div className={styles.item+' item-1'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>abyssinian-cat</p>
                            <Image id='img1' src="/imgs/cats/abyssinian-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-2'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>balinese-cat</p>
                            <Image id='img2' src="/imgs/cats/balinese-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item+' item-3'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>bengal-cat</p>
                            <Image id='img3' src="/imgs/cats/bengal-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item+' item-4'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>egyptian-mau</p>
                            <Image id='img4' src="/imgs/cats/egyptian-mau.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-5'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>maine-coon</p>
                            <Image id='img5' src="/imgs/cats/maine-coon.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-6'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>manx-cat</p>
                            <Image id='img6' src="/imgs/cats/manx-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-7'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>percian-cat</p>
                            <Image id='img7' src="/imgs/cats/percian-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-8'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>ragdoll-cat</p>
                            <Image id='img8' src="/imgs/cats/ragdoll-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-9'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>russian-blue</p>
                            <Image id='img9' src="/imgs/cats/russian-blue.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-10'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>scottish-fold</p>
                            <Image id='img10' src="/imgs/cats/scottish-fold.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-11'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>siamese-cat</p>
                            <Image id='img11' src="/imgs/cats/siamese-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-12'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>sphynx-cat</p>
                            <Image id='img12' src="/imgs/cats/sphynx-cat.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                    </div>
                </div>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className={styles.arrow} id="next" onClick={moveNext}>
                        <Image src="/imgs/right-arrow.png" alt="right-arrow" width={20} height={20} />
                    </button>
                </div>
            </div>
        </div>


        <div className='container' style={{margin:'0',padding:'10px'}}>
            <div className='row' style={{margin:'0'}}>
                <div className='col-lg-2'>
                    <Link href="/animal-detection/dogs" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <Image src="/imgs/logo2.png" alt="home"  width={200} height={140}/>
                            <h3 style={{color:'#0C0C0E'}}>DOG</h3>
                        </div>
                    </Link>
                </div>
                <div className='col-lg-8 text-center'>
                    <div id="div1" style={{width:'500px',height:'350px',border:'1px solid black',margin:'auto',display: 'flex',justifyContent: 'center',alignItems: 'center',position: 'relative',background:'lightblue'}} onDrop={drop} onDragOver={allowDrop}>
                        <h4 style={{color:'red'}}>drag and drop</h4>
                        <Image src="/imgs/drop.png" width={200} height={200} style={{position:'absolute'}} alt="drop here" />
                    </div>
                </div>
                <div className='col-lg-2'>
                    <Link href="/animal-detection/cows" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <Image src="/imgs/logo4.png" alt="logo4"  width={200} height={140}/>
                            <h3 style={{color:'#B57F2F'}}>COW</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

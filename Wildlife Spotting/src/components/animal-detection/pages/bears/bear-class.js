import React from 'react'
import styles from '../../../components/animal-detection/classic.module.css'
import Heading from '../../../components/animal-detection/heading'
import {moveNext,movePrev,allowDrop,drag,drop} from '../../../components/animal-detection/function'
import Image from 'next/image'
import Link from 'next/link'
export default function BearClass() {
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
                            <p style={{marginBottom:'0'}}>american-black-bear</p>
                            <Image id='img1' src="/imgs/bears/american-black-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-2'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>andean-bear</p>
                            <Image id='img2' src="/imgs/bears/andean-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item+' item-3'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>asian-black-bear</p>
                            <Image id='img3' src="/imgs/bears/asian-black-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item+' item-4'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>brown-bear</p>
                            <Image id='img4' src="/imgs/bears/brown-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-5'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>giant-panda</p>
                            <Image id='img5' src="/imgs/bears/giant-panda.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-6'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>polar-bear</p>
                            <Image id='img6' src="/imgs/bears/polar-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-7'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>sloth-bear</p>
                            <Image id='img7' src="/imgs/bears/sloth-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item+' item-8'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>sun-bear</p>
                            <Image id='img8' src="/imgs/bears/sun-bear.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
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
                    <Link href="/animal-detection/sheeps" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <Image src="/imgs/logo5.png" alt="logo5"  width={200} height={140}/>
                            <h3 style={{color:'#232929'}}>Sheep</h3>
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
                    <Link href="/animal-detection/elephants" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <Image src="/imgs/logo7.png" alt="logo7"  width={200} height={140}/>
                            <h3 style={{marginLeft:'70px',color:'#00052B'}}>Elephant</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

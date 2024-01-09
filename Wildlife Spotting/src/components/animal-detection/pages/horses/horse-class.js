import React from 'react'
import styles from '../../../components/animal-detection/classic.module.css'
import {moveNext,movePrev,allowDrop,drag,drop} from '../../../components/animal-detection/function'
import Heading from '../../../components/animal-detection/heading'
import Link from 'next/link'
import Image from 'next/image'
export default function HorseClass() {
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
                        <div className={styles.item +' item-1'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>andalusian</p>
                            <Image id='img1' src="/imgs/horses/andalusian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-2'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>arabian</p>
                            <Image id='img2' src="/imgs/horses/arabian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-3'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>draft-horses</p>
                            <Image id='img3' src="/imgs/horses/draft-horses.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-4'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>friesian</p>
                            <Image id='img4' src="/imgs/horses/friesian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-5'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>paint-horse</p>
                            <Image id='img5' src="/imgs/horses/paint-horse.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-6'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>palomino</p>
                            <Image id='img6' src="/imgs/horses/palomino.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-7'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>percheron</p>
                            <Image id='img7' src="/imgs/horses/percheron.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-8'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>quarter-horse</p>
                            <Image id='img8' src="/imgs/horses/quarter-horse.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-9'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>tennessee-walking-horse</p>
                            <Image id='img9' src="/imgs/horses/tennessee-walking-horse.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-10'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>thoroughbred</p>
                            <Image id='img10' src="/imgs/horses/thoroughbred.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
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
                    <Link href="/animal-detection/elephants" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}} >
                            <h3>Previous Page</h3>
                            <Image src="/imgs/logo7.png" alt="logo7"  width={200} height={140}/>
                            <h3 style={{marginLeft:'70px',color:'#00052B'}}>Elephant</h3>
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
                    <Link href="/animal-detection/zebras" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <Image src="/imgs/logo9.png" alt="logo9"  width={200} height={140}/>
                            <h3 style={{color:'#000000'}}>Zebra</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

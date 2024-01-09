import React from 'react'
import styles from '../../../components/animal-detection/classic.module.css'
import {moveNext,movePrev,allowDrop,drag,drop} from '../../../components/animal-detection/function'
import Heading from '../../../components/animal-detection/heading'
import Link from 'next/link'
import Image from 'next/image'
export default function ElephantClass() {

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
                            <p style={{marginBottom:'0'}}>african-bush-elephant</p>
                            <Image id='img1' src="/imgs/elephants/african-bush-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-2'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>african-forest-elephant</p>
                            <Image id='img2' src="/imgs/elephants/african-forest-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-3'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>baby-elephant</p>
                            <Image id='img3' src="/imgs/elephants/baby-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-4'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>elephant-with-shadow</p>
                            <Image id='img4' src="/imgs/elephants/elephant-with-shadow.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-5'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>swimming-elephant</p>
                            <Image id='img5' src="/imgs/elephants/swimming-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
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
                    <Link href="/animal-detection/bears" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <Image src="/imgs/logo6.png" alt="logo6"  width={200} height={140}/>
                            <h3 style={{color:'#99FE00'}}>Bear</h3>
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
                    <Link href="/animal-detection/horses" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <Image src="/imgs/logo8.png" alt="logo8"  width={200} height={140}/>
                            <h3 style={{color:'#005DA4'}}>Horse</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

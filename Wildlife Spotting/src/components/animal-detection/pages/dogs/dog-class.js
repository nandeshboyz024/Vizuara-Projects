import React from 'react'
import Link from 'next/link'
import styles from '../../../components/animal-detection/classic.module.css'
import {moveNext,movePrev,allowDrop,drag,drop} from '../../../components/animal-detection/function'
import Heading from '../../../components/animal-detection/heading'
import Image from 'next/image'
export default function DogClass() {

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
                        <div className={styles.item +' item-1'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>poodle</p>
                            <Image id='img1' src="/imgs/dogs/poodle.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-2'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>dalmatian</p>
                            <Image id='img2' src="/imgs/dogs/dalmatian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-3'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>basset-hound</p>
                            <Image id='img3' src="/imgs/dogs/basset-hound.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className={styles.item +' item-4'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>beagle</p>
                            <Image id='img4' src="/imgs/dogs/beagle.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-5'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>border-collie</p>
                            <Image id='img5' src="/imgs/dogs/border-collie.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-6'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>boxer</p>
                            <Image id='img6' src="/imgs/dogs/boxer.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-7'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>bull-dog</p>
                            <Image id='img7' src="/imgs/dogs/bull-dog.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-8'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>cocker-spaniel</p>
                            <Image id='img8' src="/imgs/dogs/cocker-spaniel.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-9'}style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>dachshund</p>
                            <Image id='img9' src="/imgs/dogs/dachshund.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-10'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>german-shepherd</p>
                            <Image id='img10' src="/imgs/dogs/german-shepherd.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-11'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>golden-retriever</p>
                            <Image id='img11' src="/imgs/dogs/golden-retriever.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-12'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>great-dane</p>
                            <Image id='img12' src="/imgs/dogs/great-dane.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-13'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>jack-russell-terrier</p>
                            <Image id='img13' src="/imgs/dogs/jack-russell-terrier.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-14'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>labrador-retriever</p>
                            <Image id='img14' src="/imgs/dogs/labrador-retriever.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-15'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>pug</p>
                            <Image id='img16' src="/imgs/dogs/pug.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-16'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>shetland-sheepdog</p>
                            <Image id='img17' src="/imgs/dogs/shetland-sheepdog.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className={styles.item +' item-17'} style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>siberian-husky</p>
                            <Image id='img18' src="/imgs/dogs/siberian-husky.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
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
                    <Link href="/animal-detection/birds" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <Image src="/imgs/logo1.png" alt="logo1"  width={200} height={140}/>
                            <h3 style={{color:'#E20612'}}>BIRD</h3>
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
                    <Link href="/animal-detection/cats" style={{textDecoration:'none'}}>
                        <div className={styles.item} style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <Image src="/imgs/logo3.png" alt="logo3"  width={200} height={140}/>
                            <h3 style={{color:'#DA251C'}}>CAT</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

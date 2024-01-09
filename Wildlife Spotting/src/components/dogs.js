import React from 'react'
import  './animal-detection/classic.css'
import {moveNext,movePrev,allowDrop,drag,drop} from './animal-detection/function'
import Heading from './Heading'
import {Link} from 'react-router-dom';
export default function Dogs() {

  return (
    <div>
        <Heading/>
        <div>
            <div style={{display:'flex'}}>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className='arrow' id="prev" onClick={movePrev}>
                        <img src="imgs/left-arrow.png" alt="left-arrow"  width={20} height={20}/>
                    </button>
                </div>
                <div style={{width:'90%'}}>
                    <div className='container' id="c1" style={{textAlign:'center'}}>
                        <div className='item item-1'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>poodle</p>
                            <img id='img1' src="imgs/dogs/poodle.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-2' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>dalmatian</p>
                            <img id='img2' src="imgs/dogs/dalmatian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item item-3'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>basset-hound</p>
                            <img id='img3' src="imgs/dogs/basset-hound.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item item-4'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>beagle</p>
                            <img id='img4' src="imgs/dogs/beagle.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-5'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>border-collie</p>
                            <img id='img5' src="imgs/dogs/border-collie.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-6'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>boxer</p>
                            <img id='img6' src="imgs/dogs/boxer.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-7'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>bull-dog</p>
                            <img id='img7' src="imgs/dogs/bull-dog.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-8'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>cocker-spaniel</p>
                            <img id='img8' src="imgs/dogs/cocker-spaniel.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-9'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>dachshund</p>
                            <img id='img9' src="imgs/dogs/dachshund.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-10' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>german-shepherd</p>
                            <img id='img10' src="imgs/dogs/german-shepherd.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-11' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>golden-retriever</p>
                            <img id='img11' src="imgs/dogs/golden-retriever.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-12' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>great-dane</p>
                            <img id='img12' src="imgs/dogs/great-dane.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-13' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>jack-russell-terrier</p>
                            <img id='img13' src="imgs/dogs/jack-russell-terrier.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-14' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>labrador-retriever</p>
                            <img id='img14' src="imgs/dogs/labrador-retriever.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-15' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>pug</p>
                            <img id='img16' src="imgs/dogs/pug.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-16' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>shetland-sheepdog</p>
                            <img id='img17' src="imgs/dogs/shetland-sheepdog.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-17' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>siberian-husky</p>
                            <img id='img18' src="imgs/dogs/siberian-husky.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                    </div>
                </div>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className='arrow' id="next" onClick={moveNext}>
                        <img src="imgs/right-arrow.png" alt="right-arrow" width={20} height={20} />
                    </button>
                </div>
            </div>
        </div>

        

        <div style={{padding:'20px'}}>
            <div className='row'>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/birds" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <img src="imgs/logo1.png" alt="logo1"  width={200} height={140}/>
                            <h3 style={{color:'#E20612'}}>BIRD</h3>
                        </div>
                    </Link>
                </div>
                <div className='col-lg-8 text-center'>
                    <div id="div1" style={{width:'500px',height:'350px',border:'1px solid black',margin:'auto',display: 'flex',justifyContent: 'center',alignItems: 'center',position: 'relative',background:'lightblue'}} onDrop={drop} onDragOver={allowDrop}>
                        <h4 style={{color:'red'}}>drag and drop</h4>
                        <img src="imgs/drop.png" width={200} height={200} style={{position:'absolute'}} alt="drop here" />
                    </div>
                </div>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/cats" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <img src="imgs/logo3.png" alt="logo3"  width={200} height={140}/>
                            <h3 style={{color:'#DA251C'}}>CAT</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

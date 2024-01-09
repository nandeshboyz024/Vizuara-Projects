import React from 'react'
import './animal-detection/classic.css'
import {moveNext,movePrev,allowDrop,drag,drop} from './animal-detection/function'
import {Link} from 'react-router-dom';
import Heading from './Heading'
export default function Cows() {
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
                        <div className='item  item-1' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>angus</p>
                            <img id='img1' src="imgs/cows/angus.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-2' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>ayrshire</p>
                            <img id='img2' src="imgs/cows/ayrshire.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-3' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>belgian-blue</p>
                            <img id='img3' src="imgs/cows/belgian-blue.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-4' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>dexter</p>
                            <img id='img4' src="imgs/cows/dexter.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-5' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>gelbvieh</p>
                            <img id='img5' src="imgs/cows/gelbvieh.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-6' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>hereford</p>
                            <img id='img6' src="imgs/cows/hereford.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-7' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>highland-cattle</p>
                            <img id='img7' src="imgs/cows/highland-cattle.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-8' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>holstein</p>
                            <img id='img8' src="imgs/cows/holstein.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-9' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>jersey</p>
                            <img id='img9' src="imgs/cows/jersey.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-10' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>limousin</p>
                            <img id='img10' src="imgs/cows/limousin.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-11' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>red-poll</p>
                            <img id='img11' src="imgs/cows/red-poll.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-12' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>santa-gertrudis</p>
                            <img id='img12' src="imgs/cows/santa-gertrudis.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-13' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>shorthorn</p>
                            <img id='img13' src="imgs/cows/shorthorn.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-14' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>simbrah</p>
                            <img id='img14' src="imgs/cows/simbrah.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-15' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>simmental</p>
                            <img id='img16' src="imgs/cows/simmental.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-16'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>texas-longhorn</p>
                            <img id='img17' src="imgs/cows/texas-longhorn.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
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
        <div  style={{padding:'20px'}}>
            <div className='row'>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/cats" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <img src="imgs/logo3.png" alt="logo3"  width={200} height={140}/>
                            <h3 style={{color:'#DA251C'}}>CAT</h3>
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
                    <Link to="/wildlife-spotting/sheeps" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <img src="imgs/logo5.png" alt="logo5"  width={200} height={140}/>
                            <h3 style={{color:'#232929'}}>Sheep</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

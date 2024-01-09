import React from 'react'
import Heading from './Heading'
import './animal-detection/classic.css'
import {moveNext,movePrev,allowDrop,drag,drop} from './animal-detection/function'
import {Link} from 'react-router-dom';
export default function Birds() {
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
                        <div className='item item-1' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>birds-of-prey</p>
                            <img id='img1' src="imgs/birds/birds-of-prey.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-2'style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>corvids</p>
                            <img id='img2' src="imgs/birds/corvids.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item item-3' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>gamebird</p>
                            <img id='img3' src="imgs/birds/gamebird.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item item-4' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>humming-bird</p>
                            <img id='img4' src="imgs/birds/humming-bird.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-5' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>parrot</p>
                            <img id='img5' src="imgs/birds/parrot.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-6' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>songbird</p>
                            <img id='img6' src="imgs/birds/songbird.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-7' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>waterfowl-bird</p>
                            <img id='img7' src="imgs/birds/waterfowl-bird.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-8' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>woodpacker</p>
                            <img id='img8' src="imgs/birds/woodpacker.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
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
                    <Link to="/wildlife-spotting" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Home Page</h3>
                            <img src="imgs/home.png" alt="home"  width={200} height={140}/>
                            <h3 style={{color:'#E20612'}}>Home</h3>
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
                    <Link to="/wildlife-spotting/dogs" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <img src="imgs/logo2.png" alt="logo2"  width={200} height={140}/>
                            <h3 style={{color:'#0C0C0E'}}>DOG</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

import React from 'react'
import './animal-detection/classic.css'
import {moveNext,movePrev,allowDrop,drag,drop} from './animal-detection/function'
import {Link} from 'react-router-dom';
import Heading from './Heading'
export default function Horses() {
  return (
    <div>
        <Heading/>
        <div>
            <div style={{display:'flex'}}>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className='arrow' id="prev" onClick={movePrev}>
                        <img  src="imgs/left-arrow.png" alt="left-arrow"  width={20} height={20}/>
                    </button>
                </div>
                <div style={{width:'90%'}}>
                    <div className='container' id="c1" style={{textAlign:'center'}}>
                        <div className='item item-1' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>andalusian</p>
                            <img  id='img1' src="imgs/horses/andalusian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item item-2' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>arabian</p>
                            <img  id='img2' src="imgs/horses/arabian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-3' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>draft-horses</p>
                            <img  id='img3' src="imgs/horses/draft-horses.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-4' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>friesian</p>
                            <img  id='img4' src="imgs/horses/friesian.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-5' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>paint-horse</p>
                            <img  id='img5' src="imgs/horses/paint-horse.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-6' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>palomino</p>
                            <img  id='img6' src="imgs/horses/palomino.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-7' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>percheron</p>
                            <img  id='img7' src="imgs/horses/percheron.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-8' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>quarter-horse</p>
                            <img  id='img8' src="imgs/horses/quarter-horse.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-9' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>tennessee-walking-horse</p>
                            <img  id='img9' src="imgs/horses/tennessee-walking-horse.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-10' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>thoroughbred</p>
                            <img  id='img10' src="imgs/horses/thoroughbred.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                    </div>
                </div>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className='arrow' id="next" onClick={moveNext}>
                        <img  src="imgs/right-arrow.png" alt="right-arrow" width={20} height={20} />
                    </button>
                </div>
            </div>
        </div>

        
        <div style={{padding:'20px'}}>
            <div className='row'>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/elephants" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}} >
                            <h3>Previous Page</h3>
                            <img  src="imgs/logo7.png" alt="logo7"  width={200} height={140}/>
                            <h3 style={{marginLeft:'70px',color:'#00052B'}}>Elephant</h3>
                        </div>
                    </Link>
                </div>
                <div className='col-lg-8 text-center'>
                    <div id="div1" style={{width:'500px',height:'350px',border:'1px solid black',margin:'auto',display: 'flex',justifyContent: 'center',alignItems: 'center',position: 'relative',background:'lightblue'}} onDrop={drop} onDragOver={allowDrop}>
                        <h4 style={{color:'red'}}>drag and drop</h4>
                        <img  src="imgs/drop.png" width={200} height={200} style={{position:'absolute'}} alt="drop here" />
                    </div>
                </div>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/zebras" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <img  src="imgs/logo9.png" alt="logo9"  width={200} height={140}/>
                            <h3 style={{color:'#000000'}}>Zebra</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

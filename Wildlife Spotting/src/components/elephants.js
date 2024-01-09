import React from 'react'
import './animal-detection/classic.css'
import {moveNext,movePrev,allowDrop,drag,drop} from './animal-detection/function'
import Heading from './Heading'
import {Link} from 'react-router-dom';
export default function Elephants() {

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
                            <p style={{marginBottom:'0'}}>african-bush-elephant</p>
                            <img id='img1' src="imgs/elephants/african-bush-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-2' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>african-forest-elephant</p>
                            <img id='img2' src="imgs/elephants/african-forest-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-3' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>baby-elephant</p>
                            <img id='img3' src="imgs/elephants/baby-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-4' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>elephant-with-shadow</p>
                            <img id='img4' src="imgs/elephants/elephant-with-shadow.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-5' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>swimming-elephant</p>
                            <img id='img5' src="imgs/elephants/swimming-elephant.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
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
                    <Link to="/wildlife-spotting/bears" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Previous Page</h3>
                            <img src="imgs/logo6.png" alt="logo6"  width={200} height={140}/>
                            <h3 style={{color:'#99FE00'}}>Bear</h3>
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
                    <Link to="/wildlife-spotting/horses" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <img src="imgs/logo8.png" alt="logo8"  width={200} height={140}/>
                            <h3 style={{color:'#005DA4'}}>Horse</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}

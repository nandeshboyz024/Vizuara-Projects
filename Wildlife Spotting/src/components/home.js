import React from 'react'
import './animal-detection/classic.css'
import Heading from './Heading'
import {moveNext,movePrev} from './animal-detection/function'
import {Link} from 'react-router-dom';

export default function Home() {
  return (
    <div>
        <Heading/>
        <div className='cointainer' style={{margin:'0',paddingTop:'10px'}}>
            <div className='row' style={{margin:'0'}}>
                <div className='col-lg-6'>
                    <p style={{textAlign:'justify',fontSize:'20px'}}>
                        Welcome to our wildlife haven! Here, you all discover a diverse array of inhabitants, including birds, dogs, cats, cows, sheeps, bears, elephants, horses, zebras, giraffes and humans, all coexisting in harmony. Explore the wonders of our jungle and embrace the beauty of its varied species.
                    </p>
                    <p style={{textAlign:'justify',fontSize:'20px'}}>
                        Are you feeling overwhelmed trying to identify animals? No need to worry! You are in the right place. Let AI be your guide as we explore the wonders of our jungle together.
                    </p>
                    <div className='text-center' style={{padding:'20px'}}>
                        <Link to="/wildlife-spotting/explore" style={{textDecoration:'none'}}>
                            <button type="button" style={{background:'linear-gradient(to right, #5170FF,#FF66C4)',textAlign:'center',color:'white',fontWeight:"bold",borderRadius:'10px',padding:'5px'}}>Explore Now</button>
                        </Link>
                    </div>
                </div>
                <div className='col-lg-6 text-center' >
                    <img src={process.env.PUBLIC_URL +"/imgs/family.png"} alt="family-as-wildlife" width={400} height={350}/>
                </div>
            </div>
        </div>
        <h2 style={{color:'red',marginLeft:'10px'}}>Visit To</h2>
        <div>
            <div style={{display:'flex'}}>
                <div style={{width:'5%',marginTop:'80px',textAlign:'center'}}>
                    <button className="arrow" id="prev" onClick={movePrev}>
                        <img src={process.env.PUBLIC_URL +"/imgs/left-arrow.png"} alt="left-arrow"  width={20} height={20}/>
                    </button>
                </div>
                <div style={{width:'90%'}}>
                    <div className="container" id='c1'>
                        <Link to="/wildlife-spotting/birds" style={{textDecoration:'none'}}>
                            <div className='item item-1 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo1.png"} alt="logo1"  width={200} height={140}/>
                                <h3 style={{color:'#E20612'}}>BIRD</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/dogs" style={{textDecoration:'none'}}>
                            <div className='item item-2 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo2.png"} alt="logo2"  width={200} height={140}/>
                                <h3 style={{color:'#0C0C0E'}}>DOG</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/cats" style={{textDecoration:'none'}}>
                            <div className='item item-3 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo3.png"} alt="logo3"  width={200} height={140}/>
                                <h3 style={{color:'#DA251C'}}>CAT</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/cows" style={{textDecoration:'none'}}>
                            <div className='item item-4 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo4.png"} alt="logo4"  width={200} height={140}/>
                                <h3 style={{color:'#B57F2F'}}>COW</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/sheeps" style={{textDecoration:'none'}}>
                            <div className='item item-5 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo5.png"} alt="logo5"  width={200} height={140}/>
                                <h3 style={{color:'#232929'}}>SHEEP</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/bears" style={{textDecoration:'none'}}>
                            <div className='item item-6 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo6.png"} alt="logo6"  width={200} height={140}/>
                                <h3 style={{color:'#99FE00'}}>BEAR</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/elephants" style={{textDecoration:'none'}}>
                            <div className='item item-7 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo7.png"} alt="logo7"  width={200} height={140}/>
                                <h3 style={{textAlign:'right',color:'#00052B'}}>ELEPHANT</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/horses" style={{textDecoration:'none'}}>
                            <div className='item item-8 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo8.png"} alt="logo8"  width={200} height={140}/>
                                <h3 style={{color:'#005DA4'}}>HORSE</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/zebras" style={{textDecoration:'none'}}>
                            <div className='item item-9 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo9.png"} alt="logo9"  width={200} height={140}/>
                                <h3 style={{color:'#000000'}}>ZEBRA</h3>                
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/giraffes" style={{textDecoration:'none'}}>
                            <div className='item item-10 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo10.png"} alt="logo10"  width={200} height={140}/>
                                <h3 style={{color:'#AB5613'}}>GIRAFFE</h3>
                            </div>
                        </Link>
                        <Link to="/wildlife-spotting/persons" style={{textDecoration:'none'}}>
                            <div className='item item-11 text-center' style={{margin:'10px'}}>
                                <img src={process.env.PUBLIC_URL +"/imgs/logo11.png"} alt="logo11"  width={200} height={140}/>
                                <h3 style={{color:'#6F973E'}}>HUMAN</h3>
                            </div>
                        </Link>
                    </div>
                </div>
                <div style={{width:'5%',marginTop:'80px',textAlign:'center'}}>
                    <button className="arrow" id="next" onClick={moveNext}>
                        <img src={process.env.PUBLIC_URL +"/imgs/right-arrow.png"} alt="right-arrow" width={20} height={20} />
                    </button>
                </div>
            </div>
        </div>
    </div>
  )
}

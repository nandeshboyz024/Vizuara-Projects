import React from 'react'
import Heading from '../../../components/animal-detection/heading'
import styles from '../../../components/animal-detection/global.module.css'
import Link from 'next/link'
export default function Explore() {
  return (
    <div>
        <Heading/>
        <div className='container' style={{textAlign:'justify'}}>
            <div className='row'>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Adventure Watchers</h4>
                        </div>
                        <p>Keep an eye on animals in the wild, learn how they move and live and protect those who need our help.</p>
                    </div>
                </div>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Farm Guardians</h4>
                        </div>
                        <p> Guard crops from wild visitors and find and stop bugs that bother our plants </p>
                    </div>
                </div>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Nature Detectives</h4>
                        </div>
                        <p>Check how animals change nature and make sure all kinds of living things are safe.</p>
                    </div>
                </div>
            </div>
            <div className='row'>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Wildlife Security Guards</h4>
                        </div>
                        <p> Stop animals from going where they should not and make sure our special places for animals are safe.</p>
                    </div>
                </div>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Road Rangers</h4>
                        </div>
                        <p> See if animals are on roads and help cars drive safely and avoid hitting animals.</p>
                    </div>
                </div>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Check Happiness</h4>
                        </div>
                        <p> Watch over animals in zoos or labs and make sure they are happy and healthy.</p>
                    </div>
                </div>
            </div>
            <div className='row'>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Animal Peacekeeper</h4>
                        </div>
                        <p>Find animals near homes early and tell grown-ups to keep everyone safe.</p>
                    </div>
                </div>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Hidden Camera Friends</h4>
                        </div>
                        <p>Use special cameras to spy on animals in the wild and learn about animals without bothering them.</p>
                    </div>
                </div>
                <div className={'col-lg-4 '+styles['center-div']} style={{height:'160px'}}>
                    <div className={styles.card} style={{padding:'10px'}}>
                        <div className='text-center'>
                            <h4>Rescue Heroes</h4>
                        </div>
                        <p> Find and help animals in trouble and help after big problems like floods or storms.</p>
                    </div>
                </div>
            </div>
        </div>
        <div className='text-center'>
            <Link href="/animal-detection/explore-more" style={{textDecoration:'none'}}>
                <button type="button" style={{background:'linear-gradient(to right, #5170FF,#FF66C4)',textAlign:'center',color:'white',fontWeight:"bold",borderRadius:'10px',padding:'5px'}}>Explore More</button>
            </Link>
        </div>
    </div>
  )
}

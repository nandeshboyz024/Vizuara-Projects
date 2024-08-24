import Link from "next/link"
const Errorpage = () => {
  return (
    <> <div style={{
        background: 'linear-gradient(to right, #5170FF, #FF66C4)',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
        maxWidth: '600px',
        margin: '0 auto',
        textAlign: 'center',
        fontFamily: 'Arial, sans-serif',
      }}>
        <div id ="notfound">
            <div className="notfound">
                <div className="notfound-404">
                    <h1>404</h1>
                </div>
                <h2>We are sorry, page not found!</h2>
                <p>
                    The page you are looking for might have been removed had its name changed or is temporarily unavailable.
                </p>
                <Link href="/emotion-detection" style={{color:'pink'}}>Back To Homepage</Link>
            </div>
        </div>
        </div>
    </>
  )
}

export default Errorpage

import { useState } from 'react'
import './App.css'
import LiveKitModal from './components/LikeKitModal'

function App() {
  return (
    <div className="app">
    <h1 className="absolute top-12 left-1/2 transform -translate-x-1/2 text-3xl font-bold underline text-center">
        Polaris Logistics Agent
    </h1>
      <div className="absolute top-0 left-0 w-full z-10">
        <div className="flex items-center justify-between w-full px-10">
          <img
            src="/test.png"
            alt="Qbotica Logo"
            className="h-38 w-55"
          />
          <div className="h-48 w-48" /> {/* Spacer to balance layout */}
        </div>


      </div>

      <main>
        <LiveKitModal setShowSupport={true} />
      </main>
    </div>
  )
}

export default App

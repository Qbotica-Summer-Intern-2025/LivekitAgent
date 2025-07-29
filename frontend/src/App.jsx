import { useState } from 'react'
import './App.css'
import LiveKitModal from './components/LiveKitModal'



function App() {
  return (
    <div className="app">
      <div className="fixed top-0 left-0 w-full bg-white shadow-md z-[99]">
        <div className="flex items-center h-15 px-6">
        <h1 className="font-bold text-gray-500" style={{ fontSize: '30px' }}>Qbotica</h1>
          <img
            src="/img_1.png"
            alt="Qbotica Logo"
            className="h-10"
          />
        </div>
      </div>
       {/* Main content area */}
        <h1 className="fixed top-15 left-0 w-full text-3xl font-bold underline text-center z-50 py-4">
        Logistics AI Agent
        </h1>

      <div className="mt-10">
        <LiveKitModal
          setShowSupport={true}
        />
      </div>
    </div>
  )
}

export default App;
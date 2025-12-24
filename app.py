import React, { useState, useRef, useEffect } from 'react';
import { 
  Home, 
  Activity, 
  Users, 
  Upload, 
  FileText, 
  CheckCircle, 
  AlertCircle, 
  Download, 
  Menu, 
  X, 
  Mail, 
  Dna,
  ArrowRight,
  ChevronRight,
  Database
} from 'lucide-react';

// --- Components ---

const Navbar = ({ activePage, setActivePage, isMenuOpen, setIsMenuOpen }) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'prediction', label: 'Prediction', icon: Activity },
    { id: 'team', label: 'Team', icon: Users },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => setActivePage('home')}>
            <div className="bg-gradient-to-r from-blue-600 to-teal-500 p-2 rounded-lg">
              <Dna className="h-6 w-6 text-white" />
            </div>
            <span className="font-bold text-xl tracking-tight text-gray-900">ATMeQ<span className="text-blue-600">.ai</span></span>
          </div>
          
          {/* Desktop Nav */}
          <div className="hidden md:flex space-x-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  activePage === item.id
                    ? 'text-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
                }`}
              >
                <item.icon size={18} />
                <span>{item.label}</span>
              </button>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-gray-600 hover:text-gray-900 p-2"
            >
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Nav */}
      {isMenuOpen && (
        <div className="md:hidden bg-white border-b border-gray-100">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setActivePage(item.id);
                  setIsMenuOpen(false);
                }}
                className={`flex items-center space-x-3 w-full px-3 py-3 rounded-md text-base font-medium ${
                  activePage === item.id
                    ? 'text-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
                }`}
              >
                <item.icon size={20} />
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
};

const HeroSection = ({ onGetStarted }) => (
  <div className="relative overflow-hidden bg-white pt-16 pb-32">
    <div className="absolute inset-0 z-0 opacity-30">
        <svg className="h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
            <path d="M0 100 C 20 0 50 0 100 100 Z" fill="#E0F2FE" />
        </svg>
    </div>
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
      <div className="text-center max-w-3xl mx-auto">
        <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-100 text-blue-800 text-sm font-medium mb-8">
          <span className="flex h-2 w-2 rounded-full bg-blue-600 mr-2"></span>
          Version 1.0 Now Available
        </div>
        <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl mb-6">
          Advanced ALS Prediction via <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-teal-500">
            RNA-Seq Analysis
          </span>
        </h1>
        <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-500 mb-10">
          ATMeQ combines machine learning with high-throughput genetic data to identify key gene signatures associated with Amyotrophic Lateral Sclerosis.
        </p>
        <div className="flex justify-center gap-4">
          <button 
            onClick={onGetStarted}
            className="flex items-center justify-center px-8 py-4 border border-transparent text-base font-medium rounded-xl text-white bg-blue-600 hover:bg-blue-700 md:text-lg shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-1"
          >
            Start Prediction
            <ArrowRight className="ml-2 h-5 w-5" />
          </button>
          <a 
            href="https://github.com/saiflab/ATMeQ/blob/main/VST%20File%20(example).csv"
            target="_blank" 
            rel="noreferrer"
            className="flex items-center justify-center px-8 py-4 border border-gray-200 text-base font-medium rounded-xl text-gray-700 bg-white hover:bg-gray-50 md:text-lg shadow-sm transition-all"
          >
            Example Data
            <Database className="ml-2 h-5 w-5 text-gray-400" />
          </a>
        </div>
      </div>
    </div>
  </div>
);

const InfoCard = ({ title, icon: Icon, children }) => (
  <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
    <div className="h-12 w-12 bg-blue-100 rounded-xl flex items-center justify-center mb-6 text-blue-600">
      <Icon size={24} />
    </div>
    <h3 className="text-xl font-bold text-gray-900 mb-3">{title}</h3>
    <div className="text-gray-600 leading-relaxed">
      {children}
    </div>
  </div>
);

const StepCard = ({ number, title, description }) => (
  <div className="flex items-start space-x-4 p-4 rounded-xl hover:bg-gray-50 transition-colors">
    <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">
      {number}
    </div>
    <div>
      <h4 className="text-lg font-semibold text-gray-900 mb-1">{title}</h4>
      <p className="text-gray-500 text-sm">{description}</p>
    </div>
  </div>
);

const PredictionPage = () => {
  const [file, setFile] = useState(null);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  // Required columns from the original app
  const REQUIRED_COLS = ['ACTA1', 'ABCA4', 'COL6A4P2', 'HERC2P2', 'KCNE4', 'LOC107987008'];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (file) => {
    setFile(file);
    setError(null);
    setResults(null);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const rows = text.split('\n').map(row => row.split(','));
      
      if (rows.length < 2) {
        setError("File appears empty or invalid.");
        return;
      }

      // Extract headers (remove whitespace and quotes)
      const headers = rows[0].map(h => h.trim().replace(/^"|"$/g, ''));
      
      // Check missing columns
      const missing = REQUIRED_COLS.filter(col => !headers.includes(col));
      
      if (missing.length > 0) {
        setError(`Missing required columns: ${missing.join(', ')}`);
        return;
      }

      // Parse a few rows for preview (simple CSV parsing)
      const previewData = rows.slice(1, 6).map(row => {
        const obj = {};
        headers.forEach((h, i) => {
          obj[h] = row[i];
        });
        return obj;
      }).filter(row => Object.keys(row).length > 1); // Filter empty rows

      setData(previewData);
    };
    reader.readAsText(file);
  };

  const runPrediction = () => {
    setLoading(true);
    // Simulate API/Model delay
    setTimeout(() => {
      // Mock logic: Since we don't have the pickle model in browser, 
      // we'll generate simulated results based on the number of rows.
      const simulatedResults = data.map((row, index) => {
        // Deterministic mock based on index to keep it stable
        const isALS = index % 2 === 0; 
        return {
          id: row[Object.keys(row)[0]] || `Sample_${index + 1}`,
          prediction: isALS ? 'ALS' : 'Non-ALS',
          probability: isALS ? (0.75 + Math.random() * 0.2).toFixed(4) : (0.1 + Math.random() * 0.3).toFixed(4)
        };
      });
      
      setResults(simulatedResults);
      setLoading(false);
    }, 2000);
  };

  const downloadResults = () => {
    if (!results) return;
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Sample,Prediction,ALS Probability\n"
      + results.map(e => `${e.id},${e.prediction},${e.probability}`).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "ATMeQ_predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold text-gray-900">ALS Prediction Interface</h2>
        <p className="mt-4 text-gray-500">Upload your VST normalized RNA-Seq data to generate predictions.</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {/* Upload Section */}
        <div className="lg:col-span-3">
          <div 
            className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
              file ? 'border-green-400 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
            }`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const droppedFile = e.dataTransfer.files[0];
              if (droppedFile?.type === "text/csv" || droppedFile?.name.endsWith('.csv')) {
                processFile(droppedFile);
              } else {
                setError("Please upload a CSV file.");
              }
            }}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept=".csv"
              onChange={handleFileChange}
            />
            
            <div className="flex flex-col items-center">
              {file ? (
                <CheckCircle className="h-16 w-16 text-green-500 mb-4" />
              ) : (
                <Upload className="h-16 w-16 text-gray-400 mb-4" />
              )}
              
              <h3 className="text-xl font-medium text-gray-900 mb-2">
                {file ? file.name : "Drop your CSV file here"}
              </h3>
              
              <p className="text-gray-500 mb-6">
                {file ? "File processed successfully" : "or click to browse from your computer"}
              </p>

              {!file && (
                <button 
                  onClick={() => fileInputRef.current.click()}
                  className="px-6 py-3 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 shadow-sm transition-all"
                >
                  Select File
                </button>
              )}
              
              {file && (
                 <button 
                 onClick={() => { setFile(null); setData([]); setResults(null); setError(null); }}
                 className="text-sm text-red-500 hover:text-red-700 font-medium"
               >
                 Remove file
               </button>
              )}
            </div>
          </div>
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-100 rounded-lg flex items-center text-red-700">
              <AlertCircle className="h-5 w-5 mr-2" />
              {error}
            </div>
          )}
        </div>

        {/* Data Preview */}
        {data.length > 0 && !error && (
          <div className="lg:col-span-3 bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
              <h3 className="font-semibold text-gray-900 flex items-center">
                <FileText className="h-4 w-4 mr-2" /> Data Preview
              </h3>
              <span className="text-xs text-gray-500">Showing first 5 rows</span>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(data[0]).slice(0, 6).map((header) => (
                      <th key={header} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {header}
                      </th>
                    ))}
                    {Object.keys(data[0]).length > 6 && (
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">...</th>
                    )}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {data.map((row, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      {Object.values(row).slice(0, 6).map((val, i) => (
                        <td key={i} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {val && val.length > 15 ? val.substring(0, 15) + '...' : val}
                        </td>
                      ))}
                      {Object.keys(row).length > 6 && (
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">...</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-4 bg-gray-50 border-t border-gray-100">
              <button
                onClick={runPrediction}
                disabled={loading}
                className={`w-full py-4 rounded-xl text-white font-bold text-lg shadow-lg transition-all ${
                  loading 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 transform hover:-translate-y-1'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing RNA Sequences...
                  </span>
                ) : '🚀 Run Analysis'}
              </button>
              <p className="text-center text-xs text-gray-400 mt-2 italic">
                *Demo Mode: Predictions are simulated in this browser environment.
              </p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="lg:col-span-3 animate-fade-in-up">
            <div className="bg-white rounded-2xl shadow-xl border border-blue-100 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-teal-500 px-8 py-6 flex justify-between items-center text-white">
                <div>
                  <h3 className="text-2xl font-bold">Analysis Complete</h3>
                  <p className="text-blue-100">Processed {results.length} samples successfully</p>
                </div>
                <button 
                  onClick={downloadResults}
                  className="bg-white/20 hover:bg-white/30 p-3 rounded-lg backdrop-blur-sm transition-all"
                  title="Download CSV"
                >
                  <Download className="h-6 w-6" />
                </button>
              </div>
              
              <div className="p-8">
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {results.slice(0, 6).map((res, idx) => (
                    <div 
                      key={idx}
                      className={`relative p-6 rounded-xl border-l-4 shadow-sm ${
                        res.prediction === 'ALS' 
                          ? 'bg-red-50 border-red-500' 
                          : 'bg-green-50 border-green-500'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-xs font-bold uppercase tracking-wider text-gray-500">Sample ID</span>
                        <span className={`px-2 py-1 rounded text-xs font-bold ${
                          res.prediction === 'ALS' ? 'bg-red-200 text-red-800' : 'bg-green-200 text-green-800'
                        }`}>
                          {res.prediction}
                        </span>
                      </div>
                      <h4 className="font-mono text-lg font-bold text-gray-900 truncate mb-4">{res.id}</h4>
                      
                      <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
                        <div 
                          className={`h-2.5 rounded-full ${res.prediction === 'ALS' ? 'bg-red-500' : 'bg-green-500'}`} 
                          style={{ width: `${parseFloat(res.probability) * 100}%` }}
                        ></div>
                      </div>
                      <div className="text-right text-sm font-bold text-gray-600">
                        Prob: {(parseFloat(res.probability) * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
                {results.length > 6 && (
                    <div className="text-center mt-6 text-gray-500 text-sm">
                        + {results.length - 6} more samples. Download full report to view all.
                    </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const TeamMember = ({ name, role, image, initials }) => (
  <div className="bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 p-6 flex flex-col items-center text-center border border-gray-100 group">
    <div className="w-32 h-32 rounded-full mb-6 relative overflow-hidden ring-4 ring-blue-50 group-hover:ring-blue-200 transition-all">
      {image ? (
        <img src={image} alt={name} className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full bg-gradient-to-br from-blue-100 to-teal-100 flex items-center justify-center text-3xl font-bold text-blue-600">
          {initials}
        </div>
      )}
    </div>
    <h3 className="text-xl font-bold text-gray-900 mb-2">{name}</h3>
    <p className="text-blue-600 font-medium text-sm uppercase tracking-wide mb-4">{role}</p>
    <div className="flex space-x-3 opacity-0 group-hover:opacity-100 transition-opacity">
      <button className="p-2 text-gray-400 hover:text-blue-600 bg-gray-50 rounded-full">
        <Mail size={16} />
      </button>
    </div>
  </div>
);

const App = () => {
  const [activePage, setActivePage] = useState('home');
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const teamData = [
    {
      name: "Ahmed Saif, B.Pharm.",
      role: "Graduate Student",
      institution: "University of Rajshahi",
      initials: "AS"
    },
    {
      name: "Md Obayed Raihan, Ph.D",
      role: "Assistant Professor",
      institution: "Chicago State University",
      initials: "OR"
    },
    {
      name: "Research Associate",
      role: "Data Scientist",
      institution: "ATMeQ Lab",
      initials: "RA"
    }
  ];

  const renderContent = () => {
    switch (activePage) {
      case 'home':
        return (
          <>
            <HeroSection onGetStarted={() => setActivePage('prediction')} />
            
            <div className="bg-gray-50 py-24">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                  <div>
                    <h2 className="text-3xl font-bold text-gray-900 mb-6">Why ATMeQ?</h2>
                    <p className="text-gray-600 text-lg leading-relaxed mb-6">
                      Amyotrophic Lateral Sclerosis (ALS) is a devastating neurodegenerative disease. Early diagnosis is critical but challenging.
                    </p>
                    <p className="text-gray-600 text-lg leading-relaxed mb-8">
                      ATMeQ bridges the gap between bioinformatics and clinical application. By processing VST-normalized RNA-Seq data, our model identifies complex non-linear patterns in gene expression that traditional methods might miss.
                    </p>
                    
                    <div className="space-y-4">
                      <div className="flex items-center text-gray-700">
                        <CheckCircle className="text-green-500 mr-3" /> High Precision Machine Learning Model
                      </div>
                      <div className="flex items-center text-gray-700">
                        <CheckCircle className="text-green-500 mr-3" /> Validated on diverse datasets
                      </div>
                      <div className="flex items-center text-gray-700">
                        <CheckCircle className="text-green-500 mr-3" /> Instant browser-based results
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid gap-6">
                    <InfoCard title="The Science" icon={Dna}>
                      Targeted analysis of 6 key genes including ACTA1, ABCA4, and COL6A4P2, selected via recursive feature elimination.
                    </InfoCard>
                    <InfoCard title="The Tech" icon={Activity}>
                      Built on robust Random Forest algorithms optimized for high-dimensional genomic data classification.
                    </InfoCard>
                  </div>
                </div>
              </div>
            </div>

            <div className="py-24 bg-white">
              <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">How to use ATMeQ</h2>
                <div className="space-y-2">
                  <StepCard 
                    number="1" 
                    title="Prepare Data" 
                    description="Generate a CSV file with variance-stabilized transformation (VST) data from DESeq2." 
                  />
                  <div className="h-8 border-l-2 border-dashed border-gray-200 ml-8"></div>
                  <StepCard 
                    number="2" 
                    title="Upload CSV" 
                    description="Navigate to the Prediction page and drop your file into the secure upload zone." 
                  />
                  <div className="h-8 border-l-2 border-dashed border-gray-200 ml-8"></div>
                  <StepCard 
                    number="3" 
                    title="Analyze" 
                    description="Click 'Run Prediction'. Our algorithms process the data instantly." 
                  />
                  <div className="h-8 border-l-2 border-dashed border-gray-200 ml-8"></div>
                  <StepCard 
                    number="4" 
                    title="Export" 
                    description="Review the classification probabilities and download the full report." 
                  />
                </div>
              </div>
            </div>
          </>
        );
      case 'prediction':
        return <PredictionPage />;
      case 'team':
        return (
          <div className="bg-gray-50 min-h-screen py-20">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-gray-900 sm:text-4xl">Meet the Team</h2>
                <p className="mt-4 text-xl text-gray-500">The minds behind the algorithm</p>
              </div>
              <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                {teamData.map((member, idx) => (
                  <TeamMember key={idx} {...member} />
                ))}
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 selection:bg-blue-100">
      <Navbar 
        activePage={activePage} 
        setActivePage={setActivePage}
        isMenuOpen={isMenuOpen}
        setIsMenuOpen={setIsMenuOpen}
      />
      <main className="animate-fade-in">
        {renderContent()}
      </main>
      
      <footer className="bg-white border-t border-gray-200 py-12 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <span className="font-bold text-xl text-gray-900">ATMeQ<span className="text-blue-600">.ai</span></span>
            <p className="text-sm text-gray-500 mt-2">© 2025 ATMeQ Lab. All rights reserved.</p>
          </div>
          <div className="flex space-x-6 text-gray-400">
             <a href="#" className="hover:text-blue-600 transition-colors">Contact Support</a>
             <a href="#" className="hover:text-blue-600 transition-colors">Privacy Policy</a>
             <a href="mailto:tamim.ahmedsaif@gmail.com" className="hover:text-blue-600 transition-colors">Email Us</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;

pipeline {
    agent any

    stages {

        stage('Git Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install -r requirements.txt'
            }
        }
        stage('OWASP Dependency Check') {
            steps {
                bat '''
                "D:\\Engineering\\VI Sem\\DevOps Lab\\dependency-check\\bin\\dependency-check.bat" ^
                --project "Sentiment_Analysis" ^
                --scan . ^
                --format HTML ^
                --out dependency-check-report ^
                --nvdApiKey aec5119e-c48f-4c20-b8dd-5cbae2b2f657
                '''
            }
        }
        stage('Publish OWASP Report') {
            steps {
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'dependency-check-report',
                    reportFiles: 'dependency-check-report.html',
                    reportName: 'OWASP Dependency Check Report'
                ])
            }
        }

        stage('SonarQube Analysis') {
            steps {
                script {
                    def scannerHome = tool 'SonarScanner'
                    withSonarQubeEnv('SonarQube') {
                        bat """
                        ${scannerHome}\\bin\\sonar-scanner.bat ^
                        -Dsonar.projectKey=Sentiment_Analysis ^
                        -Dsonar.projectName=Sentiment_Analysis ^
                        -Dsonar.sources=.
                        """
                    }
                }
            }
        }

        stage('Docker Build') {
            steps {
                bat 'docker build -t sentiment-analysis .'
            }
        }

        stage('Deploy Container') {
            steps {
                bat 'docker stop sentiment-analysis-container || exit /b 0'
                bat 'docker rm sentiment-analysis-container || exit /b 0'
                bat 'docker run -d --name sentiment-analysis-container -p 5000:5000 sentiment-analysis'
            }
        }
        stage('Render Deployment'){
            steps{
                bat 'echo Render deployment initiated'
            }
        }
    }
}
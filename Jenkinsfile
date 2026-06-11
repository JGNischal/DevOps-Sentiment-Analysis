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
                bat 'pip install -r requirements.txt'
            }
        }
stage('OWASP Dependency Check') {
    steps {
        bat '''
        C:\\Tools\\dependency-check\\bin\\dependency-check.bat ^
        --project "ML-Pro" ^
        --scan . ^
        --format HTML ^
        --out dependency-check-report ^
        --nvdApiKey d24f4b4f-d0dd-4a08-8214-9bd6352cd42e
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
                        -Dsonar.projectKey=ml-pro-app ^
                        -Dsonar.projectName=ml-pro-app ^
                        -Dsonar.sources=.
                        """
                    }
                }
            }
        }

   stage('Docker Build') {
    steps {
        bat 'docker build -t ml-pro-app .'
    }
}

stage('Deploy Container') {
    steps {
        bat 'docker stop ml-pro-app-container || exit /b 0'
        bat 'docker rm ml-pro-app-container || exit /b 0'
        bat 'docker run -d --name ml-pro-app-container -p 5050:5050 ml-pro-app'
    }
}
    }
}
package com.example.demoappmlhd

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.audio.audioclassifier.AudioClassifier
import com.google.mediapipe.tasks.audio.audioclassifier.AudioClassifier.AudioClassifierOptions
import com.google.mediapipe.tasks.audio.audioclassifier.AudioClassifierResult
import com.google.mediapipe.tasks.audio.core.RunningMode
//import com.google.mediapipe.tasks.components.containers.Category
//import com.google.mediapipe.tasks.components.containers.Classifications
import com.google.mediapipe.tasks.core.BaseOptions
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import kotlin.collections.get
import kotlin.text.toInt
import kotlin.times

class MainActivity : AppCompatActivity() {

    private lateinit var resultsTextView: TextView
    private lateinit var startButton: Button
    private lateinit var progressBar: ProgressBar

    private var audioClassifier: AudioClassifier? = null
    private lateinit var backgroundExecutor: ScheduledExecutorService

    private val modelPath = "model.tflite"
    private val probabilityThreshold = 0.2f

    companion object {
        private const val TAG = "AudioClassifier"
        private const val REQUEST_RECORD_AUDIO = 1337
        private const val MAX_RESULTS = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout)
        enableEdgeToEdge()

        resultsTextView = findViewById(R.id.textViewResults)
        startButton = findViewById(R.id.buttonStart)
        progressBar = findViewById(R.id.progressBar)

        startButton.setOnClickListener {
            Log.d(TAG, "Button clicked")
            if (audioClassifier != null) {
                stopAudioClassification()
            } else {
                startAudioClassification()
            }
        }

        backgroundExecutor = Executors.newSingleThreadScheduledExecutor()
    }

    private fun startAudioClassification() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO
            )
        } else {
            backgroundExecutor.execute {
                try {
                    val baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build()

                    val options = AudioClassifierOptions.builder()
                        .setBaseOptions(baseOptions)
                        .setMaxResults(MAX_RESULTS)
                        .setScoreThreshold(probabilityThreshold)
                        .setRunningMode(RunningMode.AUDIO_STREAM)
                        .setResultListener { result -> onResult(result) }
                        .setErrorListener(this::onError)
                        .build()

                    audioClassifier = AudioClassifier.createFromOptions(this@MainActivity, options)

                    val audioRecord = audioClassifier!!.createAudioRecord()
                    audioRecord.startRecording()

                    runOnUiThread {
                        startButton.text = getString(R.string.stop_recognizing)
                        progressBar.visibility = View.VISIBLE
                        resultsTextView.text = getString(R.string.listening)
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Error during classificator creation", e)
                    runOnUiThread {
                        Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                        progressBar.visibility = View.GONE
                        startButton.text = getString(R.string.start_recognition)
                    }
                }
            }
        }
    }

    private fun stopAudioClassification() {
        backgroundExecutor.execute {
            audioClassifier?.close()
            audioClassifier = null

            runOnUiThread {
                startButton.text = getString(R.string.start_recognition)
                progressBar.visibility = View.GONE
                resultsTextView.text = getString(R.string.press_start_recognition_to_begin)
            }
        }
    }


    private fun onResult(result: AudioClassifierResult) {
        runOnUiThread {
            if (result.classificationResults().isNotEmpty()) {
                val classificationResult = result.classificationResults()[0]

                if (classificationResult.classifications().isNotEmpty()) {
                    val classifications = classificationResult.classifications()[0]

                    if (classifications.categories().isNotEmpty()) {
                        val resultsStr = buildString {
                            append("Commands found:\n\n")
                            for (category in classifications.categories()) {
                                val command = category.categoryName() // or category.displayName()
                                val confidence = (category.score() * 100).toInt()
                                append("  - ${command.replaceFirstChar { it.uppercase() }}: ${confidence}%\n")
                            }
                        }
                        resultsTextView.text = resultsStr
                    } else {
                        resultsTextView.text = getString(R.string.failed_recognition)
                    }
                } else {
                    resultsTextView.text = getString(R.string.failed_recognition)
                }
            } else {
                resultsTextView.text = getString(R.string.failed_recognition)
            }
        }
    }

    private fun onError(error: RuntimeException) {
        Log.e(TAG, "Audio classification error: ${error.message}")
        runOnUiThread {
            Toast.makeText(this, "Error: ${error.message}", Toast.LENGTH_SHORT).show()
            if(audioClassifier != null) {
                stopAudioClassification()
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startAudioClassification()
        } else {
            Toast.makeText(this, "Required RECORD_AUDIO permission to use the app.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onPause() {
        super.onPause()
        if (audioClassifier != null) {
            stopAudioClassification()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::backgroundExecutor.isInitialized) {
            backgroundExecutor.shutdownNow()
        }
    }
}
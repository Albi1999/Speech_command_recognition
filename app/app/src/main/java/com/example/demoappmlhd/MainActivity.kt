package com.example.demoappmlhd

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresPermission
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import be.tarsos.dsp.mfcc.MFCC
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var resultsTextView: TextView
    private lateinit var startButton: Button
    private lateinit var progressBar: ProgressBar

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private var bufferSizeInBytes = 0

    private lateinit var tfliteInterpreter: Interpreter
    //private val modelPath = "model_cnn_transformer_refined_new.tflite"
    //private val modelPath = "model_cnn_transformer_refined.tflite"
    private val modelPath = "model_cnn_transformer.tflite"
    private val labelsPath = "labels.txt"
    private lateinit var labels: List<String>

    companion object {
        private const val TAG = "AudioClassifierManual"
        private const val REQUEST_RECORD_AUDIO = 1337
        private const val RECORDING_LENGTH_IN_SECONDS = 3
        private const val MAX_RESULTS = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout)
        enableEdgeToEdge()

        resultsTextView = findViewById(R.id.textViewResults)
        startButton = findViewById(R.id.buttonStart)
        progressBar = findViewById(R.id.progressBar)

        try {
            tfliteInterpreter = Interpreter(loadModelFile(modelPath))
            labels = loadLabels(labelsPath)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing TensorFlow Lite.", e)
            Toast.makeText(this, "Failed to initialize model: ${e.message}", Toast.LENGTH_LONG).show()
        }

        startButton.setOnClickListener {
            if (isRecording) {
                Log.d(TAG, "Already recording, ignoring click.")
                return@setOnClickListener
            } else {
                checkPermissionsAndStartRecording()
            }
        }
    }

    private fun checkPermissionsAndStartRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        } else {
            startRecordingAndProcessing()
        }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startRecordingAndProcessing() {
        if (isRecording) return
        isRecording = true

        bufferSizeInBytes = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        val recordingBufferSize = sampleRate * RECORDING_LENGTH_IN_SECONDS * 2

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSizeInBytes
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord could not be initialized.")
            isRecording = false
            return
        }

        runOnUiThread {
            startButton.isEnabled = false
            startButton.text = getString(R.string.listening)
            progressBar.visibility = View.VISIBLE
            resultsTextView.text = ""
        }

        audioRecord?.startRecording()

        thread {
            val audioData = ShortArray(recordingBufferSize / 2)
            audioRecord?.read(audioData, 0, audioData.size)

            isRecording = false
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            runOnUiThread {
                progressBar.visibility = View.GONE
                resultsTextView.text = getString(R.string.processing)
            }

            try {
                val spectrogram = computeLogMelSpectrogram(audioData)
                val inputBuffer = convertSpectrogramToByteBuffer(spectrogram)
                val outputBuffer = Array(1) { FloatArray(labels.size) }
                tfliteInterpreter.run(inputBuffer, outputBuffer)
                val resultsStr = decodeOutput(outputBuffer[0])

                runOnUiThread {
                    resultsTextView.text = resultsStr
                    startButton.isEnabled = true
                    startButton.text = getString(R.string.start_recognition)
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error during processing or inference", e)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun computeLogMelSpectrogram(audioData: ShortArray): Array<FloatArray> {
        val floatAudioData = FloatArray(audioData.size) { i ->
            audioData[i].toFloat() / Short.MAX_VALUE.toFloat()
        }

        val nFFT = 400
        val hopLength = 160
        val nMels = 40
        val nFrames = 101

        val mfcc = MFCC(nFFT, sampleRate.toFloat(), 20, nMels, 300.0f, 8000.0f)

        val melSpectrogram = Array(nMels) { FloatArray(nFrames) }

        for (frame in 0 until nFrames) {
            val start = frame * hopLength
            if (start + nFFT > floatAudioData.size) break

            val audioFrame = floatAudioData.sliceArray(start until start + nFFT)

            val magnitudeSpectrum = mfcc.magnitudeSpectrum(audioFrame)

            val melFiltered = mfcc.melFilter(magnitudeSpectrum, mfcc.centerFrequencies)

            val logMel = mfcc.nonLinearTransformation(melFiltered)

            for (i in 0 until nMels) {
                if (i < logMel.size) {
                    melSpectrogram[i][frame] = logMel[i]
                }
            }
        }

        val flattened = melSpectrogram.flatMap { it.asIterable() }
        val mean = flattened.average().toFloat()
        val stdDev = sqrt(flattened.map { (it - mean) * (it - mean) }.average()).toFloat()

        for (i in 0 until nMels) {
            for (j in 0 until nFrames) {
                melSpectrogram[i][j] = (melSpectrogram[i][j] - mean) / (stdDev + 1e-9f)
            }
        }

        return melSpectrogram
    }

    private fun convertSpectrogramToByteBuffer(spectrogram: Array<FloatArray>): ByteBuffer {
        val inputShape = tfliteInterpreter.getInputTensor(0).shape() // [1, 40, 101, 1]
        val byteBuffer = ByteBuffer.allocateDirect(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (i in 0 until inputShape[1]) {
            for (j in 0 until inputShape[2]) {
                byteBuffer.putFloat(spectrogram[i][j])
            }
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun decodeOutput(outputArray: FloatArray): String {
        val topResults = outputArray
            .mapIndexed { index, confidence -> Pair(labels[index], confidence) }
            .sortedByDescending { it.second }
            .take(MAX_RESULTS)

        return buildString {
            append("Commands found:\n\n")
            topResults.forEach { (label, confidence) ->
                val confidencePercent = (confidence * 100).toInt()
                append("  - ${label.replaceFirstChar { it.uppercase() }}: $confidencePercent%\n")
            }
        }
    }

    private fun loadModelFile(filePath: String): ByteBuffer {
        val assetFileDescriptor = assets.openFd(filePath)
        return assetFileDescriptor.createInputStream().use { inputStream ->
            val fileChannel = inputStream.channel
            fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        }
    }

    private fun loadLabels(filePath: String): List<String> {
        return assets.open(filePath).bufferedReader().readLines()
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecordingAndProcessing()
        } else {
            Toast.makeText(this, "Permission denied. App cannot work.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter.close()
    }
}
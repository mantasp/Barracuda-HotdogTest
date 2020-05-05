using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
//using Barracuda;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.UI;

public class RunTest : MonoBehaviour
{
	public float epsilon = 1e-3f;
	public NNModel srcModel;
	public Texture2D inputImage;
	public TextAsset labelsAsset;
	public int inputResolutionY = 224;
	public int inputResolutionX = 224;
	public Material preprocessMaterial;
	
	public bool useGPU = true;
	
	public Text text;
	public RawImage displayImage;

	public int repeatExecution = 1;

	private Model model;
	private IWorker engine;
	private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
	private string[] labels;
	
	private float averageDt;
	private float rawAverageDt;

	// Use this for initialization
	IEnumerator Start ()
	{
		Application.targetFrameRate = 60;
		
		labels = labelsAsset.text.Split('\n');
		model = ModelLoader.Load(srcModel, false);
		engine = WorkerFactory.CreateWorker(useGPU ? WorkerFactory.Type.ComputePrecompiled : WorkerFactory.Type.CSharp, model, false);
		
		var input = new Tensor(PrepareTextureForInput(inputImage, !useGPU), 3);
		
		inputs["input"] = input;

		yield return null;

		StartCoroutine(RunInference());
	}
	
	IEnumerator RunInference ()
	{
		// Skip frame before starting
		yield return null;
		displayImage.texture = inputImage;

		while (repeatExecution-- > 0)
		{
			try
			{
				var start = Time.realtimeSinceStartup;
				
				Profiler.BeginSample("Schedule execution");
				engine.Execute(inputs);
				Profiler.EndSample();
				
				Profiler.BeginSample("Fetch execution results");
				var output = engine.PeekOutput();
				Profiler.EndSample();

				var res = output.ArgMax()[0];
				var end = Time.realtimeSinceStartup;
				var label = labels[res];

				if (label.Contains("hotdog") && Mathf.Abs(output[res] - 0.978f) < epsilon)
				{
					text.color = Color.green;
					text.text = $"Success: {labels[res]} {output[res] * 100}%";
				}
				else
				{
					text.color = Color.red;
					text.text = $"Failed: {labels[res]} {output[res] * 100}%";
				}

				UpdateAverage(end - start);
				Debug.Log($"frametime = {(end - start)*1000f}ms, average = {averageDt * 1000}ms");
			
				
			}
			catch (Exception e)
			{
				Debug.Log($"Exception happened {e}");
				//throw;
			}
			yield return null;
		}
	}
	
	// Scale image to target resolution and remap data from 0..1 to -1..1 on GPU
	Texture PrepareTextureForInput(Texture2D src, bool needsCPUcopy)
	{
		var targetRT = RenderTexture.GetTemporary(inputResolutionX, inputResolutionY, 0, RenderTextureFormat.ARGB32);
		RenderTexture.active = targetRT;
		Graphics.Blit(src, targetRT, preprocessMaterial);

		if (!needsCPUcopy)
			return targetRT;
		
		var  result = new Texture2D(targetRT.width, targetRT.height);
		result.ReadPixels(new Rect(0,0, targetRT.width, targetRT.height), 0, 0);
		result.Apply();

		return result;
	}

	private void UpdateAverage(float newValue)
	{
		rawAverageDt = rawAverageDt * 0.9f + 0.1f * newValue;
		
		// Drop spikes above 20%
		if (newValue < 1.2f * rawAverageDt)
		{
			averageDt = averageDt * 0.9f + 0.1f * newValue;
		}
	}

	private void OnDestroy()
	{
		engine?.Dispose();

		foreach (var key in inputs.Keys)
		{
			inputs[key].Dispose();
		}
		
		inputs.Clear();
	}
}

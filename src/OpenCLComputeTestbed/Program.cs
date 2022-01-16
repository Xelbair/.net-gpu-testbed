using OpenTK.Compute.OpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;

namespace OpenCLComputeTestbed
{
    internal class Program
    {
        private static readonly int LIST_SIZE = 1024;
        private static readonly int LOCAL_ITEM_SIZE = 64;
        private static readonly ConsoleColor DEFAULT_FOREGROUND_COLOR = Console.ForegroundColor;
        private static string OPENCL_SOURCE = "ComputePrograms\\VectorAddition.cl";
        static void Main(string[] args)
        {
            var a = CreateArray(x => x);
            var b = CreateArray(x => x);
            int[] c = new int[a.Length];
            var programSource = File.ReadAllText(OPENCL_SOURCE);

            //build context
            var deviceIDs = GetDevices();
            var context = BuildContext(deviceIDs);
            //create queue
            var queue = BuildCommandQueue(context, deviceIDs[0]);
            //create buffers
            var aBuffer = CreateBuffer(context, MemoryFlags.ReadOnly, a.Length, sizeof(int));
            var bBuffer = CreateBuffer(context, MemoryFlags.ReadOnly, b.Length, sizeof(int));
            var cBuffer = CreateBuffer(context, MemoryFlags.WriteOnly, c.Length, sizeof(int));
            // queue write for buffers
            Console.WriteLine("Writing data to GPU Buffers");
            EnqueueBufferWrite(queue, aBuffer, a);
            EnqueueBufferWrite(queue, bBuffer, b);

            var (kernel, program) = BuildKernel(context, programSource, deviceIDs);

            var bindings = new Dictionary<uint, CLBuffer>
            {
                { 0, aBuffer },
                { 1, bBuffer },
                { 2, cBuffer },
            };
            BindBuffersToKernel(kernel, bindings);
            //execute kernel
            ExecuteKernel(queue, kernel);

            ReadBuffer(queue, cBuffer, c);

            //Test Results - wouldn't make sense but alas this is a simple and test kernel
            TestKernel(a, b, c, (a, b) => a+b);

            //cleanup of resources bound on GPU
            Console.WriteLine("Cleanup");
            CL.Flush(queue);
            CL.Finish(queue);
            CL.ReleaseKernel(kernel);
            CL.ReleaseProgram(program);
            CL.ReleaseMemoryObject(aBuffer);
            CL.ReleaseMemoryObject(bBuffer);
            CL.ReleaseMemoryObject(cBuffer);
            CL.ReleaseCommandQueue(queue);
            CL.ReleaseContext(context);
            Console.WriteLine("Cleanup Finished");

        }

        private static void TestKernel(int[] a, int[] b, int[] c, Func<int, int, int> kernel)
        {
            Console.Write("TESTING RESULT: ");
            bool ok = CompareArrays(a, b, c, kernel);
            WritePrettyResult(ok ? CLResultCode.Success : CLResultCode.InvalidValue);
        }

        private static void ReadBuffer(CLCommandQueue queue, CLBuffer buffer, int[] resultArray)
        {
            Console.Write("reading buffer to array: ");
            var ret = CL.EnqueueReadBuffer(queue, buffer, true, new UIntPtr(0), resultArray, null, out _);
            WritePrettyResult(ret);
        }

        private static void ExecuteKernel(CLCommandQueue queue, CLKernel kernel)
        {
            Console.Write("Executing kernel: ");
            var globalItemSize = new UIntPtr((uint)LIST_SIZE);
            var localitemSize = new UIntPtr((uint)LOCAL_ITEM_SIZE);
            var ret = CL.EnqueueNDRangeKernel(queue, kernel, 1, null, new UIntPtr[] { globalItemSize }, new UIntPtr[] { localitemSize }, 0, null, out _);
            WritePrettyResult(ret);

        }

        private static void BindBuffersToKernel(CLKernel kernel, Dictionary<uint, CLBuffer> bufferMap)
        {
            foreach (var item in bufferMap)
            {
                Console.Write($"Binding buffer to argument {item.Key}: ");
                var ret = CL.SetKernelArg(kernel, item.Key, item.Value);
                WritePrettyResult(ret);
            }
        }

        private static (CLKernel kernel, CLProgram program) BuildKernel(CLContext context, string programSource, CLDevice[] deviceIDs)
        {
            Console.Write("Loading program into the GPU memory: ");
            var program = CL.CreateProgramWithSource(context, programSource, out var ret);
            WritePrettyResult(ret);

            Console.Write("Compiling program on GPU: ");
            ret = CL.BuildProgram(program, 1, deviceIDs, null, IntPtr.Zero, IntPtr.Zero);
            WritePrettyResult(ret);

            Console.Write("Creating Kernel: ");
            var kernel = CL.CreateKernel(program, "vector_add", out ret);
            WritePrettyResult(ret);

            //we need to also return the program because we need to clean it up later on to free resources
            return (kernel, program);
        }

        private static void EnqueueBufferWrite(CLCommandQueue queue, CLBuffer buffer, int[] data)
        {
            Console.Write("Enqueueing write: ");
            var ret = CL.EnqueueWriteBuffer(queue, buffer, true, UIntPtr.Zero, data, null, out _);
            WritePrettyResult(ret);
        }

        private static CLBuffer CreateBuffer(CLContext context, MemoryFlags flags, int elementCount, int elementSize)
        {
            Console.Write("Creating Buffer: ");
            var sizePtr = new UIntPtr((uint)(elementCount * elementSize));
            var buffer = CL.CreateBuffer(context, flags, sizePtr, IntPtr.Zero, out CLResultCode ret);
            WritePrettyResult(ret);
            return buffer;
        }
        private static CLBuffer CreateBuffer(CLContext context, MemoryFlags flags, int[] array)
        {
            return CL.CreateBuffer(context, flags, array, out var cLResultCode);
        }

        private static CLCommandQueue BuildCommandQueue(CLContext context, CLDevice deviceID)
        {
            Console.Write("Creating Queue: ");
            var queue = CL.CreateCommandQueueWithProperties(context, deviceID, IntPtr.Zero, out var ret);
            WritePrettyResult(ret);
            return queue;
        }

        private static CLContext BuildContext(CLDevice[] deviceIDs)
        {
            Console.Write("Creating Context: ");
            var context = CL.CreateContext(IntPtr.Zero, deviceIDs, IntPtr.Zero, IntPtr.Zero, out var ret);
            WritePrettyResult(ret);
            return context;
        }

        private static CLDevice[] GetDevices()
        {
            var ret = CL.GetPlatformIds(out CLPlatform[] platformId);
            ret = CL.GetDeviceIds(platformId[0], DeviceType.Gpu, out CLDevice[] deviceIDs);
            return deviceIDs;
        }

        /// <summary>
        /// Helper method that just logs result of OpenCL operations
        /// </summary>
        /// <param name="ret"></param>
        static void WritePrettyResult(CLResultCode ret)
        {
            Console.ForegroundColor = ret == CLResultCode.Success ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine(ret);
            Console.ForegroundColor = DEFAULT_FOREGROUND_COLOR;
        }

        /// <summary>
        /// Test method that computes the same function as OpenCL Kernel, passed as lambda, and compares the result with kernel's.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="comparer">function equivalent to kernel used to compute <paramref name="c"/></param>
        /// <returns></returns>
        private static bool CompareArrays<T>(T[] a, T[] b, T[] c, Func<T, T, T> comparer) where T : IEquatable<T>
        {
            for (int i = 0; i < LIST_SIZE; i++)
            {
                var testOk = comparer(a[i], b[i]).Equals(c[i]);
                if (!testOk) return false;
            }
            return true;
        }

        /// <summary>
        /// Helper method that populates array with elements based on index
        /// </summary>
        /// <param name="builder"></param>
        /// <returns></returns>
        static int[] CreateArray(Func<int, int> builder)
        {
            int[] result = new int[LIST_SIZE];
            for (int i = 0; i < LIST_SIZE; i++)
            {
                result[i] = builder(i);
            }
            return result;
        }
    }
}

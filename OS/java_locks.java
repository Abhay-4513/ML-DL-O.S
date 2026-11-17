import java.util.concurrent.Semaphore;

class ProducerConsumer {

    static final int SIZE = 5;

    static int[] buffer = new int[SIZE];
    static int in = 0, out = 0;

    // Semaphores
    static Semaphore empty = new Semaphore(SIZE);
    static Semaphore full = new Semaphore(0);
    static Semaphore mutex = new Semaphore(1);

    // Producer thread
    static class Producer extends Thread {

        int id;

        Producer(int id) {
            this.id = id;
        }

        public void run() {

            for (int i = 1; i <= 5; i++) {

                try {
                    empty.acquire();
                    mutex.acquire();

                    int item = id * 100 + i;

                    buffer[in] = item;

                    System.out.println(
                        "Producer " + id +
                        " produced: " + item
                    );

                    in = (in + 1) % SIZE;

                    mutex.release();
                    full.release();

                    Thread.sleep(200);

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    // Consumer thread
    static class Consumer extends Thread {

        int id;

        Consumer(int id) {
            this.id = id;
        }

        public void run() {

            for (int i = 1; i <= 5; i++) {

                try {
                    full.acquire();
                    mutex.acquire();

                    int item = buffer[out];

                    System.out.println(
                        "Consumer " + id +
                        " consumed: " + item
                    );

                    out = (out + 1) % SIZE;

                    mutex.release();
                    empty.release();

                    Thread.sleep(300);

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    public static void main(String[] args)
            throws InterruptedException {

        Producer p1 = new Producer(1);
        Producer p2 = new Producer(2);

        Consumer c1 = new Consumer(1);
        Consumer c2 = new Consumer(2);

        p1.start();
        p2.start();

        c1.start();
        c2.start();

        p1.join();
        p2.join();

        c1.join();
        c2.join();

        System.out.println("All items produced and consumed.");
    }
}

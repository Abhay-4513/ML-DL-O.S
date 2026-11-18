#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5
#define ITEMS_PER_PRODUCER 5
#define TOTAL_ITEMS (2 * ITEMS_PER_PRODUCER) // 2 producers * 5 items = 10 items total

int buffer[BUFFER_SIZE];
int in = 0;  // Index for producer to insert
int out = 0; // Index for consumer to remove

// Semaphores
sem_t empty; // Counts empty slots in the buffer
sem_t full;  // Counts filled slots in the buffer
sem_t mutex; // Binary semaphore for mutual exclusion (lock)

// Producer Thread Function
void *producer(void *arg) {
    int id = *((int *)arg);
    for (int i = 1; i <= ITEMS_PER_PRODUCER; i++) {
        int item = id * 100 + i; // Unique item value

        // 1. Wait for an empty slot
        sem_wait(&empty);

        // 2. Enter critical section (lock mutex)
        sem_wait(&mutex);

        // Critical Section: Add item to buffer
        buffer[in] = item;
        printf("-> Producer %d produced item %d at buffer index %d\n", id, item, in);
        in = (in + 1) % BUFFER_SIZE;

        // 3. Exit critical section (unlock mutex)
        sem_post(&mutex);

        // 4. Signal that a new item is available (increment full)
        sem_post(&full);

        // Sleep briefly to simulate work
        usleep(100000); 
    }
    pthread_exit(NULL);
}

// Consumer Thread Function
void *consumer(void *arg) {
    int id = *((int *)arg);
    // Each consumer consumes 5 items across the lifetime (total 10 items consumed by 2 consumers)
    for (int i = 1; i <= ITEMS_PER_PRODUCER; i++) {
        // 1. Wait for a filled slot
        sem_wait(&full);

        // 2. Enter critical section (lock mutex)
        sem_wait(&mutex);

        // Critical Section: Remove item from buffer
        int item = buffer[out];
        printf("<- Consumer %d consumed item %d from buffer index %d\n", id, item, out);
        out = (out + 1) % BUFFER_SIZE;

        // 3. Exit critical section (unlock mutex)
        sem_post(&mutex);

        // 4. Signal that an empty slot is available (increment empty)
        sem_post(&empty);

        // Sleep briefly to simulate work
        usleep(150000);
    }
    pthread_exit(NULL);
}

int main() {
    pthread_t prod[2], cons[2];
    int prod_ids[2] = {1, 2};
    int cons_ids[2] = {1, 2};

    // Initialize semaphores
    sem_init(&empty, 0, BUFFER_SIZE); // Initially 5 empty slots
    sem_init(&full, 0, 0);            // Initially 0 filled slots
    sem_init(&mutex, 0, 1);           // Binary semaphore initialized to 1

    printf("=== PRODUCER-CONSUMER PROBLEM (Buffer Size = %d) ===\n", BUFFER_SIZE);

    // Create Producer threads
    for (int i = 0; i < 2; i++) {
        pthread_create(&prod[i], NULL, producer, &prod_ids[i]);
    }

    // Create Consumer threads
    for (int i = 0; i < 2; i++) {
        pthread_create(&cons[i], NULL, consumer, &cons_ids[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < 2; i++) {
        pthread_join(prod[i], NULL);
    }
    for (int i = 0; i < 2; i++) {
        pthread_join(cons[i], NULL);
    }

    // Destroy semaphores
    sem_destroy(&empty);
    sem_destroy(&full);
    sem_destroy(&mutex);

    printf("=== ALL ITEMS PRODUCED AND CONSUMED SUCCESSFULLY ===\n");
    return 0;
}

import com.datastax.spark.connector._
import java.io.{ByteArrayOutputStream, ByteArrayInputStream}
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

case class CuckooServiceName(service_name: String, sha256: String)
case class CuckooLabel(sha256: String, source_tags: String)
case class CuckooServiceNameAndLabel(sha256: String, service_name: String, source_tags: String)
case class CuckooSha256(sha256: String, service_name: String, results: Array[Byte])
case class CuckooJoin(sha256: String, service_name: String, results: String, source_tags: String)

def unzip(x: Array[Byte]) : String = {
    val inputStream = new GZIPInputStream(new ByteArrayInputStream(x));
    return scala.io.Source.fromInputStream(inputStream).mkString;
}

val cuckoo_service_name_rdd = sc.cassandraTable[CuckooServiceName](KEYSPACE, SERVICE_NAME_TABLE)
                            .where("service_name=?", "CUCKOO")
                            .keyBy(x => (x.sha256));
val cuckoo_label_rdd = sc.cassandraTable[CuckooLabel](KEYSPACE, OBJECTS_TABLE)
                     .keyBy(x => (x.sha256));
val cuckoo_service_name_and_label_rdd = cuckoo_service_name_rdd.join(cuckoo_label_rdd)
                                      .map(x =>(new CuckooServiceNameAndLabel(x._1, x._2._1.service_name, x._2._2.source_tags)))
                                      .keyBy(x => (x.sha256, x.service_name));

val cuckoo_sha256_rdd = sc.cassandraTable[CuckooSha256](KEYSPACE, SHA256_TABLE)
                      .where("service_name=?", "CUCKOO")
                      .keyBy(x => (x.sha256, x.service_name));
val cuckoo_join = cuckoo_service_name_and_label_rdd.join(cuckoo_sha256_rdd)
                .map(x => (new CuckooJoin(x._1._1, x._1._2, unzip(x._2._2.results), x._2._1.source_tags)))
                .distinct();

cuckoo_join.toDF().write.format("parquet").save("cuckoo.df");

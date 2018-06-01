import com.datastax.spark.connector._
import java.io.{ByteArrayOutputStream, ByteArrayInputStream}
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

case class PeinfoServiceName(service_name: String, sha256: String)
case class PeinfoLabel(sha256: String, source_tags: String)
case class PeinfoServiceNameAndLabel(sha256: String, service_name: String, source_tags: String)
case class PeinfoSha256(sha256: String, service_name: String, results: Array[Byte])
case class PeinfoJoin(sha256: String, service_name: String, results: String, source_tags: String)

def unzip(x: Array[Byte]) : String = {
    val inputStream = new GZIPInputStream(new ByteArrayInputStream(x));
    return scala.io.Source.fromInputStream(inputStream).mkString;
}

val peinfo_service_name_rdd = sc.cassandraTable[PeinfoServiceName](KEYSPACE, SERVICE_NAME_TABLE)
                            .where("service_name=?", "peinfo")
                            .keyBy(x => (x.sha256));
val peinfo_label_rdd = sc.cassandraTable[PeinfoLabel](KEYSPACE, OBJECTS_TABLE)
                     .keyBy(x => (x.sha256));
val peinfo_service_name_and_label_rdd = peinfo_service_name_rdd.join(peinfo_label_rdd)
                                      .map(x =>(new PeinfoServiceNameAndLabel(x._1, x._2._1.service_name, x._2._2.source_tags)))
                                      .keyBy(x => (x.sha256, x.service_name));

val peinfo_sha256_rdd = sc.cassandraTable[PeinfoSha256](KEYSPACE, SHA256_TABLE)
                      .where("service_name=?", "peinfo")
                      .keyBy(x => (x.sha256, x.service_name));
val peinfo_join = peinfo_service_name_and_label_rdd.join(peinfo_sha256_rdd)
                .map(x => (new PeinfoJoin(x._1._1, x._1._2, unzip(x._2._2.results), x._2._1.source_tags)))
                .distinct();

peinfo_join.toDF().write.format("parquet").save("peinfo.df");

